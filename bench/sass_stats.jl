# Instruction counts of the kernels in a SASS dump (added 2026-10-05; `CODE.md`, "The
# right-hand side on an H200"). It reads what `bench/rhs_lab.jl` writes with `sass=1`
# or in mode `baseline`, as `CUDA.code_sass` and `CUDA.@device_code_sass` print it.
# For each kernel it reports:
#
# - the instructions — static, so a loop's body counts once;
# - the highest register named;
# - the FP64 arithmetic (`DFMA`, `DMUL`, `DADD`);
# - the most frequent opcodes — `STL`/`LDL` are spills, `LDG` global loads;
# - the call targets.
#
# It needs nothing but Julia:
#
#     julia bench/sass_stats.jl out/sass-pkg-inl-N32.txt
using Printf

function sass_stats(io::IO, path::AbstractString)
    s = read(path, String)
    heads = collect(eachmatch(r"//-+ \.text\.(\S+) -+", s))
    for (n, m) in enumerate(heads)
        lo = m.offset + ncodeunits(m.match)
        hi = n < length(heads) ? heads[n + 1].offset - 1 : ncodeunits(s)
        body = SubString(s, lo, prevind(s, hi + 1))
        name = m.captures[1]
        short = match(r"_Z\d+(\w+?_)\d", name)
        short = short === nothing ? first(name, 40) : short.captures[1]
        ops = Dict{String,Int}()
        ninstr = 0
        for im in eachmatch(r"^\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)\b"m, body)
            op = first(split(im.captures[1], '.'))
            ops[op] = get(ops, op, 0) + 1
            ninstr += 1
        end
        regs = maximum((parse(Int, r.captures[1]) for r in eachmatch(r"\bR(\d+)\b", body));
                       init=-1) + 1
        calls = Dict{String,Int}()
        for c in eachmatch(r"CALL\.REL\.NOINC `\(\$(?:[^$]*\$)?([^)]+)\)", body)
            calls[c.captures[1]] = get(calls, c.captures[1], 0) + 1
        end
        fp64 = get(ops, "DFMA", 0) + get(ops, "DMUL", 0) + get(ops, "DADD", 0)
        @printf(io, "%-30s instr %6d  maxreg %3d  fp64 %d\n", short, ninstr, regs, fp64)
        top = first(sort(collect(ops); by=p -> -p.second), 24)
        println(io, "   ", join(("$k:$v" for (k, v) in top), "  "))
        isempty(calls) || println(io, "   calls: ", join(("$k:$v" for (k, v) in calls), "  "))
    end
    return nothing
end

for path in ARGS
    println("== ", path)
    sass_stats(stdout, path)
end
