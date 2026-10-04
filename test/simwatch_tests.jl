# SimWatch status files (added 2026-10-02): `src/simwatch.jl`. Host-side and
# cheap; what is claimed is what a viewer and a long run rely on — a file
# SimWatch can read, a write that never stops the run, and an
# `update_interval` that does not call a slow run stale.

using Test
using TOML
using TreeGeneralizedHarmonic

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
end
