using Test
using Logging
using StorageGraphs

@testset "Initial Conditions" begin

TEST_DIR = (@__DIR__)*"/../output"

if isdir(TEST_DIR)
    rm(TEST_DIR, force=true, recursive=true)
end

params = PhysicalParameters()
r = (@__DIR__)*"/../output/classical"

@testset "No files" begin
    q0, p0 = @test_logs((:debug, "Generated 2 initial conditions."),
        min_level=Logging.Debug, match_mode=:any,
        initial_conditions(10., alg=PoincareRand(n=2), root=r))
    for i in axes(q0, 1)
        @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
    end
end

@testset "BSON Backup" begin
    g = initialize(r)
    savechanges(g, backup=true)
    rm(r*"/graph.jls")
    q0, p0 = initial_conditions(10., alg=PoincareRand(n=2), root=r)
    for i in axes(q0, 1)
        @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
    end
end

@testset "Basic save / load" begin
    algs = [PoincareUniform(n=3,m=3), InscribedCircle(n=2, m=2)]
    counts = [2, 4]
    for i in eachindex(algs)
        q0, p0 = @test_logs((:debug, "Generated $(counts[i]) initial conditions."),
            min_level=Logging.Debug, match_mode=:any,
            initial_conditions(10., alg=algs[i], root=r))
        @test counts[i] == size(q0, 1)
        for i in axes(q0, 1)
            @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
        end
    end

    algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
    counts = [2, 2, 4]
    for i in eachindex(algs)
        q0, p0 = initial_conditions(10., alg=algs[i], root=r)
        @test counts[i] == size(q0, 1)
        for i in axes(q0, 1)
            @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
        end
    end
end

# @testset "Recompute" begin
#     algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
#     counts = [2, 2, 4]
#     for i ∈ eachindex(algs)
#         q0, p0 = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
#             (:debug, "Incompatible initial conditions. Generating new conditions."),
#             (:debug, "Generated $(counts[i]) initial conditions."),
#             (:debug, "Deleting rows"),
#             (:debug, "Appending with missing"),
#             min_level=Logging.Debug, match_mode=:all,
#             initial_conditions(10., alg=algs[i], recompute=true, root=r))
#         @test counts[i] == size(q0, 1)
#         for i in axes(q0, 1)
#             @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
#         end
#     end
# end
#


end
