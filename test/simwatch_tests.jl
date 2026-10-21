# SimWatch status files (added 2026-10-02): `src/simwatch.jl`. Host-side and
# cheap; what is claimed is what a viewer and a long run rely on — a file
# SimWatch can read, a write that never stops the run, and an
# `update_interval` that does not call a slow run stale.

using Test
using TOML
using TreeGeneralizedHarmonic

# The thumbnails `test/octant_runs.jl` draws (SVG from 2026-10-21).
include(joinpath(@__DIR__, "simwatch_plots.jl"))

# A small check that an SVG is well formed: every element closed, in order,
# and nothing but one `<svg>` root. Julia has no XML parser in its standard
# library, and the thumbnails are a handful of element kinds.
function sw_wellformed(svg)
    stack = String[]
    roots = 0
    body = replace(svg, r"<\?xml[^>]*\?>" => "")
    for m in eachmatch(r"<(/?)([A-Za-z][A-Za-z0-9]*)((?:[^>\"]|\"[^\"]*\")*?)(/?)>", body)
        close, name, selfclose = m.captures[1] == "/", m.captures[2], m.captures[4] == "/"
        if close
            (isempty(stack) || pop!(stack) != name) && return false
        elseif !selfclose
            isempty(stack) && (roots += 1)
            push!(stack, name)
        else
            isempty(stack) && return false
        end
    end
    # Every `<` and `>` is part of a tag, and the text between holds no `&`
    # that is not an entity.
    count(==('<'), body) == count(==('>'), body) || return false
    occursin(r"&(?![a-z]+;|#[0-9]+;)", body) && return false
    return isempty(stack) && roots == 1
end

@testset verbose = true "SimWatch status files" begin
    @testset "the document holds the known keys, and problem tables extend them" begin
        # Guards a table named like a known one being overwritten instead of
        # extended, and a `nothing` or `missing` written as a placeholder.
        doc = simwatch_document(; name="run", status="running", message="a\nb",
                                progress=(time=1.5, time_end=missing, iteration=7),
                                extra=Dict("progress" => Dict("chunk" => 3),
                                           "constraints" => (ham_l2=NaN, mom_l2=nothing)),
                                black_holes=[(name="BH", irreducible_mass=1.0,
                                              position=(0.0, 0.0, 0.0), found=true)])
        @test doc["message"] == "a"
        @test doc["progress"] == Dict("time" => 1.5, "iteration" => 7, "chunk" => 3)
        @test isnan(doc["constraints"]["ham_l2"]) && !haskey(doc["constraints"], "mom_l2")
        @test doc["black_holes"][1]["position"] == [0.0, 0.0, 0.0]
        dir = mktempdir()
        @test write_simwatch(dir, doc)
        back = TOML.parsefile(joinpath(dir, "simwatch.toml"))
        @test back["name"] == "run" && back["progress"]["chunk"] == 3
        @test isnan(back["constraints"]["ham_l2"])
        @test !isfile(joinpath(dir, "simwatch.toml.tmp"))
    end

    @testset "a write that fails returns false and never throws" begin
        # Guards the rule that a status file must not stop a run: a directory
        # that cannot be created (here, below a regular file).
        f = joinpath(mktempdir(), "file")
        write(f, "x")
        @test write_simwatch(joinpath(f, "run"), Dict("a" => 1)) == false
        sw = SimWatchWriter(joinpath(f, "run"); name="r")
        @test (@test_logs (:warn,) simwatch_update!(sw; force=true)) == false
        @test simwatch_update!(sw; force=true) == false        # warned once only
    end

    @testset "the rate limit, and an update_interval that follows the calls" begin
        # Guards a run whose calls are farther apart than the minimum spacing
        # being shown as stale between them: the reported interval is the
        # larger of the minimum and 1.5 times the observed spacing, and the
        # startup interval before there is a spacing.
        dir = mktempdir()
        sw = SimWatchWriter(dir; name="r", interval=60, startup_interval=900)
        read_interval() = TOML.parsefile(joinpath(dir, "simwatch.toml"))["update_interval"]
        @test simwatch_update!(sw; status="starting", time=0.0)
        @test read_interval() == 900
        @test !simwatch_update!(sw; time=0.5)                  # within the minimum
        sw.last_call -= 400                                    # as if 400 s had passed
        sw.last_write -= 400
        @test simwatch_update!(sw; time=1.0)
        @test read_interval() ≈ 600 atol = 1
        doc = TOML.parsefile(joinpath(dir, "simwatch.toml"))
        @test doc["progress"]["time_start"] == 0.0 && doc["status"] == "running"
        @test simwatch_finish!(sw; status="stopped", time=2.0)
        @test TOML.parsefile(joinpath(dir, "simwatch.toml"))["status"] == "stopped"
    end

    @testset "the history window: padded, trimmed, rounded, and refilled by hand" begin
        # Guards the sparklines' series falling out of step with `time` (a
        # series that skips a point must be padded, not shortened), a window
        # that grows without bound, and an all-`NaN` series being written.
        dir = mktempdir()
        sw = SimWatchWriter(dir; history_length=3, summary=["constraints.ham_l2"])
        for (i, t) in enumerate(1.0:5.0)
            h = Dict{String,Any}("a" => 0.1234567891 * t)
            i == 2 && (h["b"] = 7)
            simwatch_update!(sw; force=true, time=t, history=h)
        end
        doc = TOML.parsefile(joinpath(dir, "simwatch.toml"))
        @test doc["summary"] == ["constraints.ham_l2"]
        @test doc["history"]["time"] == [3.0, 4.0, 5.0]
        @test doc["history"]["a"] ≈ [0.370370, 0.493827, 0.617284] atol = 1e-6
        @test !haskey(doc["history"], "b")
        simwatch_update!(sw; force=true, history=Dict("a" => 1.0))   # no time: no point
        @test TOML.parsefile(joinpath(dir, "simwatch.toml"))["history"]["time"] ==
              [3.0, 4.0, 5.0]
        sw2 = SimWatchWriter(dir; history_length=3)
        simwatch_history!(sw2, 1.0, Dict("a" => 2.0))                # a restart's refill
        simwatch_history!(sw2, 2.0, Dict("c" => 3.0))
        @test simwatch_update!(sw2; force=true)
        h = TOML.parsefile(joinpath(dir, "simwatch.toml"))["history"]
        @test h["time"] == [1.0, 2.0] && h["c"][2] == 3.0 && isnan(h["c"][1])
        @test h["a"][1] == 2.0 && isnan(h["a"][2])
    end

    @testset "group, fraction and a slurm table reach the file (SimWatch 1.3.0)" begin
        # Guards the keys SimWatch's own writer writes and this one did not
        # until 2026-10-21: a study's `group`, the `fraction` of a run not
        # measured in simulation time, and `slurm` keys from the caller.
        dir = mktempdir()
        sw = SimWatchWriter(dir; name="r", group="study")
        @test simwatch_update!(sw; force=true, fraction=0.25,
                               slurm=(next_job_id="9",))
        doc = TOML.parsefile(joinpath(dir, "simwatch.toml"))
        @test doc["group"] == "study" && doc["progress"]["fraction"] == 0.25
        @test doc["slurm"]["next_job_id"] == "9"
    end

    @testset "a chain of jobs: the earlier ones are previous_job_ids" begin
        # Guards the status file being the job's rather than the
        # simulation's: a job that starts where another job's file is must
        # list that job (and a `next_job_id` that never wrote) as earlier
        # ones, oldest first, and a restart within one job must not list
        # itself.
        dir = mktempdir()
        write_simwatch(dir, simwatch_document(; name="r", status="stopped",
                                              slurm=(job_id="100",
                                                     previous_job_ids=["90"],
                                                     next_job_id="101")))
        withenv("SLURM_JOB_ID" => "102") do
            sw = SimWatchWriter(dir; name="r")
            @test sw.previous_job_ids == ["90", "100", "101"]
            @test simwatch_update!(sw; force=true)
            sl = TOML.parsefile(joinpath(dir, "simwatch.toml"))["slurm"]
            @test sl["job_id"] == "102" && sl["previous_job_ids"] == ["90", "100", "101"]
            @test !haskey(sl, "next_job_id")
            @test SimWatchWriter(dir; name="r").previous_job_ids == ["90", "100", "101"]
        end
        @test SimWatchWriter(mktempdir()).previous_job_ids == String[]
        # The batch jobs' shell helper (`bench/simwatch.sh`, a copy of
        # SimWatch's `writers/simwatch.sh`) on this writer's file: queued as
        # job 103, then job 103 starts.
        sh = joinpath(@__DIR__, "..", "bench", "simwatch.sh")
        if Sys.which("bash") !== nothing && Sys.which("awk") !== nothing
            run(`bash -c ". \"$sh\" && simwatch_queued \"$dir\" \"\" 103"`)
            q = TOML.parsefile(joinpath(dir, "simwatch.toml"))
            @test q["status"] == "queued" && q["slurm"]["next_job_id"] == "103"
            @test q["name"] == "r"
            withenv("SLURM_JOB_ID" => "103") do
                @test SimWatchWriter(dir).previous_job_ids == ["90", "100", "101", "102"]
            end
            run(`bash -c ". \"$sh\" && simwatch_mark \"$dir\" failed \"exit code 1\""`)
            m = TOML.parsefile(joinpath(dir, "simwatch.toml"))
            @test m["status"] == "failed" && m["message"] == "exit code 1"
        end
    end

    @testset "the thumbnails are SVG as SimWatch 1.3.0 asks" begin
        # Guards a thumbnail SimWatch cannot show or cannot adapt: not well
        # formed, referring to another file, painting its own background
        # (SimWatch adapts a transparent image with `currentColor` ink to its
        # user's terminal; a painted one would be the "dark on light" this
        # replaced), larger than its limits, or written in place. And a plot
        # that cannot be drawn must return `false`, never throw.
        dir = mktempdir()
        path = joinpath(dir, "plots", "ham.svg")
        n = 5000                                  # more points than are drawn
        t = collect(range(0, 50; length=n))
        a = [1e-6 * exp(x / 10) for x in t]
        # Every fifth point is drawn (1, 6, 11, …): `b` has a point alone at
        # the start, then three gaps.
        b = [k in (1001, 2001, 3001) ? NaN : k == 6 ? -1.0 : 1e-9 * (1 + x)
             for (k, x) in enumerate(t)]
        @test simwatch_plot(path, t, [a, b])
        svg = read(path, String)
        @test !isfile(path * ".tmp")
        @test startswith(svg, "<?xml") && sw_wellformed(svg)
        root = match(r"<svg [^>]*>", svg).match
        @test occursin("xmlns=\"http://www.w3.org/2000/svg\"", root)
        @test occursin("width=\"400\"", root) && occursin("height=\"300\"", root)
        @test occursin("viewBox=\"0 0 400 300\"", root)
        @test filesize(path) < 1024^2 ÷ 4          # SimWatch refuses 1 MiB
        @test !occursin("<image", svg) && !occursin("href", svg)
        @test !occursin("<rect", svg) && !occursin("background", svg)
        @test occursin("stroke=\"currentColor\"", svg) && occursin("fill=\"currentColor\"", svg)
        # The ink is `currentColor`, `none` or the palette, nothing else.
        colours = Set(m.captures[1] for m in eachmatch(r"(?:fill|stroke)=\"([^\"]*)\"", svg))
        @test colours ⊆ Set(["currentColor", "none", SW_PALETTE...])
        @test SW_PALETTE[1] in colours && SW_PALETTE[2] in colours
        # Each series is broken where a value is missing, and thinned.
        @test count("<polyline", svg) == 1 + 4 && count("<circle", svg) == 1
        @test length(eachmatch(r"\d,\d", svg) |> collect) ≤ 2 * 1000 + 10
        @test occursin(">10<tspan", svg) && occursin("−", svg)
        if Sys.which("xmllint") !== nothing
            @test success(`xmllint --noout $path`)
        end
        # A status file naming it: the path relative to the run directory.
        write_simwatch(dir, simwatch_document(; name="r",
                                              images=[(file="plots/ham.svg", title="ℋ",
                                                       description="log scale")]))
        img = only(TOML.parsefile(joinpath(dir, "simwatch.toml"))["images"])
        @test img["file"] == "plots/ham.svg" && isfile(joinpath(dir, img["file"]))
        # Nothing to draw, or nowhere to draw it.
        @test simwatch_plot(joinpath(dir, "none.svg"), Float64[], [Float64[]]) == false
        @test simwatch_plot(joinpath(dir, "neg.svg"), [0.0, 1.0], [[0.0, -1.0]]) == false
        @test !isfile(joinpath(dir, "none.svg")) && !isfile(joinpath(dir, "neg.svg"))
        f = joinpath(dir, "file")
        write(f, "x")
        @test (@test_logs (:warn,) simwatch_plot(joinpath(f, "a.svg"), t, [a])) == false
    end
end
