using Test
using Logging
using DataFrames, CSV

@testset "Lyapunov" begin

TEST_DIR = (@__DIR__)*"/../output"

if isdir(TEST_DIR)
    rm(TEST_DIR, force=true, recursive=true)
end

params = PhysicalParameters()
r = (@__DIR__)*"/../output/classical"

@testset "No files" begin
    @test_logs((:debug, "Generated 2 initial conditions."),
        min_level=Logging.Debug, match_mode=:any,
        λmap(10., ic_alg=PoincareRand(n=2), root=r))
end

@testset "Basic save / load" begin
    λalgs = [DynSys()]
    for λa in setdiff(λalgs, [DynSys()])
        λmap(10., ic_alg=PoincareRand(n=2), alg=λa, root=r)
    end
    for λa in λalgs
        algs = [PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
        counts = [2, 4]
        for i ∈ eachindex(algs)
            λs = @test_logs(
                (:debug, "Generated $(counts[i]) initial conditions."),
                min_level=Logging.Debug, match_mode=:any,
                λmap(10., ic_alg=algs[i], alg=λa, root=r))

            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.3)
        end

        algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
        counts = [2, 2, 4]
        for i ∈ eachindex(algs)
            λs = λmap(10., ic_alg=algs[i], alg=λa, root=r)
            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.3)
        end
    end
end

# @testset "Recompute" begin
#     λalgs = [DynSys()]
#     algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
#     counts = [2, 2, 4]
#     for λa in λalgs
#         for i ∈ eachindex(algs)
#             λs = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
#                 (:debug, "Incompatible initial conditions. Generating new conditions."),
#                 (:debug, "Generated $(counts[i]) initial conditions."),
#                 (:debug, "Deleting rows"),
#                 (:debug, "Appending with missing"),
#                 (:debug, "Initial conditions compat"),
#                 (:debug, "Stored values compat"),
#                 # when recomputing the initial conditinos the whole line is deleted
#                 (:debug, "Incompatible values. Computing new values."),
#                 (:debug, "Fixing missing values"),
#                 (:debug, "Selected compatible initial conditions"),
#                 (:debug, "Updating copy"),
#                 (:debug, "Updating in-place"),
#                 (:debug, "Deleting rows"),
#                 min_level=Logging.Debug, match_mode=:all,
#                 λmap(10., ic_alg=algs[i], alg=λa, ic_recompute=true, root=r))
#             @test length(λs) == counts[i]
#             @test all(0 .< λs .< 0.3)
#
#             λs = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
#                 (:debug, "Loading compatible initial conditions."),
#                 (:debug, "Initial conditions compat"),
#                 (:debug, "Stored values compat"),
#                 (:debug, "Incompatible values. Computing new values."),
#                 (:debug, "Selected compatible initial conditions"),
#                 (:debug, "Updating copy"),
#                 (:debug, "Updating in-place"),
#                 (:debug, "Deleting rows"),
#                 min_level=Logging.Debug, match_mode=:all,
#                 λmap(10., ic_alg=algs[i], alg=λa, recompute=true, root=r))
#             @test length(λs) == counts[i]
#             @test all(0 .< λs .< 0.3)
#         end
#     end
# end

@testset "Other DB operations" begin
    λalgs = [DynSys(T=12000.)]
    algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
    counts = [2, 2, 4]
    for λa in λalgs
        for i ∈ eachindex(algs)
            λs = λmap(10., ic_alg=algs[i], alg=λa, root=r)
            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.3)
        end
    end

    for λa in λalgs
        for i ∈ eachindex(algs)
            λs = @test_logs((:debug, "Generated $(counts[i]) initial conditions."),
                min_level=Logging.Debug, match_mode=:any,
                λmap(20., ic_alg=algs[i], alg=λa, root=r))
            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.4)
        end
    end

end

end
