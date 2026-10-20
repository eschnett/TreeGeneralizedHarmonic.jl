# Thumbnails for SimWatch (added 2026-10-19): time series from a run's CSV
# drawn as small PNGs, which `simwatch.toml`'s `[[images]]` point at and
# SimWatch shows in its detail view (`FORMAT.md`: at most 512×512, about
# 400×300 is best, replaced atomically before the status file is rewritten).
# `FORMAT.md` suggests a dark background; these are dark on light (Erik's
# choice, 2026-10-19).
#
# Included by `test/octant_runs.jl`. Plain Julia with no dependencies, so that
# the run's environment does not grow a plotting package: a PNG of stored
# (uncompressed) deflate blocks, a log-scale line plot, and a 3×5 digit font
# for the axis labels. Nothing here may stop a run — `simwatch_plot` returns
# `false` instead of throwing.

const SW_BG = (0xfc, 0xfc, 0xfa)
const SW_AXIS = (0x40, 0x40, 0x48)
const SW_GRID = (0xdc, 0xdc, 0xe0)
const SW_PALETTE = [(0xd5, 0x5e, 0x00), (0x00, 0x72, 0xb2), (0x00, 0x9e, 0x73),
                    (0x96, 0x3c, 0xaa), (0x1e, 0x1e, 0x1e), (0x8c, 0x8c, 0x8c),
                    (0xc8, 0x1e, 0x1e)]
const SW_PALETTE_NAMES = ["orange", "blue", "green", "purple", "black", "grey", "red"]

# --- PNG -----------------------------------------------------------------------

const _SW_CRC_TABLE = let t = zeros(UInt32, 256)
    for n in 0:255
        c = UInt32(n)
        for _ in 1:8
            c = isodd(c) ? 0xedb88320 ⊻ (c >> 1) : c >> 1
        end
        t[n + 1] = c
    end
    t
end

function _sw_crc32(bytes)
    c = 0xffffffff
    for b in bytes
        c = _SW_CRC_TABLE[((c ⊻ b) & 0xff) + 1] ⊻ (c >> 8)
    end
    return c ⊻ 0xffffffff
end

function _sw_adler32(bytes)
    a, b = UInt32(1), UInt32(0)
    for x in bytes
        a = (a + x) % 65521
        b = (b + a) % 65521
    end
    return (b << 16) | a
end

_sw_be32(x) = UInt8[(x >> 24) & 0xff, (x >> 16) & 0xff, (x >> 8) & 0xff, x & 0xff]

function _sw_chunk(io, kind::String, data::Vector{UInt8})
    write(io, _sw_be32(UInt32(length(data))))
    body = vcat(Vector{UInt8}(kind), data)
    write(io, body)
    write(io, _sw_be32(_sw_crc32(body)))
    return nothing
end

# `img[y, x]` an RGB triple, row 1 at the top.
function _sw_write_png(path, img::Matrix{NTuple{3,UInt8}})
    hgt, wid = size(img)
    raw = UInt8[]
    sizehint!(raw, hgt * (3wid + 1))
    for y in 1:hgt
        push!(raw, 0x00)                    # filter: none
        for x in 1:wid
            append!(raw, img[y, x])
        end
    end
    # zlib with stored deflate blocks of at most 65535 bytes.
    z = UInt8[0x78, 0x01]
    n = length(raw)
    i = 1
    while true
        len = min(65535, n - i + 1)
        last = i + len > n
        push!(z, last ? 0x01 : 0x00)
        append!(z, UInt8[len & 0xff, len >> 8, ~len & 0xff, (~len >> 8) & 0xff])
        append!(z, view(raw, i:(i + len - 1)))
        i += len
        last && break
    end
    append!(z, _sw_be32(_sw_adler32(raw)))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
        _sw_chunk(io, "IHDR", vcat(_sw_be32(UInt32(wid)), _sw_be32(UInt32(hgt)),
                                   UInt8[8, 2, 0, 0, 0]))
        _sw_chunk(io, "IDAT", z)
        _sw_chunk(io, "IEND", UInt8[])
    end
    mv(tmp, path; force=true)
    return nothing
end

# --- drawing -------------------------------------------------------------------

# A 3×5 font: each glyph five rows of three bits, the top row first.
const _SW_FONT = Dict{Char,NTuple{5,UInt8}}(
    '0' => (7, 5, 5, 5, 7), '1' => (2, 6, 2, 2, 7), '2' => (7, 1, 7, 4, 7),
    '3' => (7, 1, 7, 1, 7), '4' => (5, 5, 7, 1, 1), '5' => (7, 4, 7, 1, 7),
    '6' => (7, 4, 7, 5, 7), '7' => (7, 1, 1, 1, 1), '8' => (7, 5, 7, 5, 7),
    '9' => (7, 5, 7, 1, 7), '-' => (0, 0, 7, 0, 0), '.' => (0, 0, 0, 0, 2),
    'e' => (0, 7, 7, 4, 7), 't' => (2, 7, 2, 2, 3), 'M' => (5, 7, 7, 5, 5),
    ' ' => (0, 0, 0, 0, 0))

function _sw_text!(img, x0, y0, s, color; scale=2)
    x = x0
    for ch in s
        g = get(_SW_FONT, ch, _SW_FONT[' '])
        for r in 1:5, c in 1:3
            (g[r] >> (3 - c)) & 1 == 1 || continue
            for dy in 0:(scale - 1), dx in 0:(scale - 1)
                yy, xx = y0 + (r - 1) * scale + dy, x + (c - 1) * scale + dx
                checkbounds(Bool, img, yy, xx) && (img[yy, xx] = color)
            end
        end
        x += 4scale
    end
    return nothing
end

function _sw_line!(img, x0, y0, x1, y1, color, box)
    xlo, xhi, ylo, yhi = box
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for k in 0:ceil(Int, n)
        x = round(Int, x0 + (x1 - x0) * k / n)
        y = round(Int, y0 + (y1 - y0) * k / n)
        for (dx, dy) in ((0, 0), (1, 0), (0, 1))
            xx, yy = x + dx, y + dy
            xlo ≤ xx ≤ xhi && ylo ≤ yy ≤ yhi && (img[yy, xx] = color)
        end
    end
    return nothing
end

"""
    simwatch_plot(path, t, series; width = 400, height = 300) -> Bool

A log-scale line plot of the vectors in `series` against `t`, one colour of
`SW_PALETTE` each in order, written to `path` as a PNG: decades labelled on
the left, the time axis below. Non-positive and non-finite values are left
out of their line. Returns whether it worked; it never throws.
"""
function simwatch_plot(path, t::AbstractVector, series; width=400, height=300)
    try
        img = fill(SW_BG, height, width)
        xl, xr, yt, yb = 34, width - 8, 8, height - 22
        box = (xl, xr, yt, yb)
        vals = [log10(v) for s in series for v in s if isfinite(v) && v > 0]
        (length(t) ≥ 1 && !isempty(vals)) || return false
        lo, hi = floor(minimum(vals)), ceil(maximum(vals))
        hi == lo && (hi += 1)
        t0, t1 = Float64(minimum(t)), Float64(maximum(t))
        t1 == t0 && (t1 = t0 + 1)
        px(tt) = xl + (xr - xl) * (tt - t0) / (t1 - t0)
        py(lv) = yb - (yb - yt) * (lv - lo) / (hi - lo)
        # The decades, labelled with their exponent where there is room.
        step = max(1, ceil(Int, (hi - lo) / 10))
        for d in Int(lo):Int(hi)
            y = round(Int, py(d))
            for x in xl:xr
                img[y, x] = SW_GRID
            end
            (d - Int(lo)) % step == 0 && _sw_text!(img, 2, y - 4, string(d), SW_AXIS)
        end
        # The time axis: four ticks, labelled.
        for k in 0:4
            tt = t0 + k * (t1 - t0) / 4
            x = round(Int, px(tt))
            for y in yt:yb
                img[y, x] = SW_GRID
            end
            lbl = string(round(tt; digits=tt < 10 ? 1 : 0))
            _sw_text!(img, clamp(x - 8, 0, width - 4 * 2 * length(lbl)), yb + 6, lbl,
                      SW_AXIS)
        end
        for x in xl:xr
            img[yb, x] = SW_AXIS
        end
        for y in yt:yb
            img[y, xl] = SW_AXIS
        end
        for (i, s) in enumerate(series)
            color = SW_PALETTE[mod1(i, length(SW_PALETTE))]
            prev = nothing
            for (tt, v) in zip(t, s)
                if !(isfinite(v) && v > 0)
                    prev = nothing
                    continue
                end
                pt = (px(Float64(tt)), py(log10(v)))
                if prev === nothing
                    _sw_line!(img, pt..., pt..., color, box)
                else
                    _sw_line!(img, prev..., pt..., color, box)
                end
                prev = pt
            end
        end
        mkpath(dirname(path))
        _sw_write_png(path, img)
        return true
    catch err
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
