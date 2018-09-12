using Test
using Logging
using DataFrames, CSV

@testset "Lyapunov" begin

TEST_DIR = "output/classical/B0.55-D0.4/"

if isdir(TEST_DIR)
    rm(TEST_DIR, force=true, recursive=true)
end

params = PhysicalParameters()

@testset "No files" begin
    @test_logs((:debug, "No initial conditions file found. Generating new conditions."),
        (:debug, "Generated 2 initial conditions."),
        (:debug, "Initial conditions compat"),
        (:debug, "Stored values compat"),
        (:debug, "Incompatible values. Computing new values."),
         # compatibility check adds missing values and the columns have the type `Vector{Missing}`
        (:debug, "Fixed column types"),
        (:debug, "Selected compatible initial conditions"),
        (:debug, "Updating copy"),
        (:debug, "Updating in-place"),
        (:debug, "Deleting rows"),
        min_level=Logging.Debug, match_mode=:all,
        λmap(10., ic_alg=PoincareRand(n=2)))
end

@testset "Basic save / load" begin
    λalgs = [DynSys()]
    for λa in setdiff(λalgs, [DynSys()])
        λmap(10., ic_alg=PoincareRand(n=2), alg=λa)
    end
    for λa in λalgs
        algs = [PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
        counts = [2, 4]
        for i ∈ eachindex(algs)
            λs = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
                (:debug, "Incompatible initial conditions. Generating new conditions."),
                (:debug, "Generated $(counts[i]) initial conditions."),
                (:debug, "Appending with missing"),
                (:debug, "Initial conditions compat"),
                (:debug, "Stored values compat"),
                (:debug, "Incompatible values. Computing new values."),
                (:debug, "Fixing missing values"),
                (:debug, "Selected compatible initial conditions"),
                (:debug, "Updating copy"),
                (:debug, "Updating in-place"),
                (:debug, "Deleting rows"),
                min_level=Logging.Debug, match_mode=:all,
                λmap(10., ic_alg=algs[i], alg=λa))

            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.3)
        end

        algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
        counts = [2, 2, 4]
        for i ∈ eachindex(algs)
            λs = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
                (:debug, "Loading compatible initial conditions."),
                (:debug, "Initial conditions compat"),
                (:debug, "Stored values compat"),
                (:debug, "Loading compatible values."),
                min_level=Logging.Debug, match_mode=:all,
                λmap(10., ic_alg=algs[i], alg=λa))
            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.3)
        end
    end
end

@testset "Recompute" begin
    λalgs = [DynSys()]
    algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
    counts = [2, 2, 4]
    for λa in λalgs
        for i ∈ eachindex(algs)
            λs = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
                (:debug, "Incompatible initial conditions. Generating new conditions."),
                (:debug, "Generated $(counts[i]) initial conditions."),
                (:debug, "Deleting rows"),
                (:debug, "Appending with missing"),
                (:debug, "Initial conditions compat"),
                (:debug, "Stored values compat"),
                # when recomputing the initial conditinos the whole line is deleted
                (:debug, "Incompatible values. Computing new values."),
                (:debug, "Fixing missing values"),
                (:debug, "Selected compatible initial conditions"),
                (:debug, "Updating copy"),
                (:debug, "Updating in-place"),
                (:debug, "Deleting rows"),
                min_level=Logging.Debug, match_mode=:all,
                λmap(10., ic_alg=algs[i], alg=λa, ic_recompute=true))
            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.3)

            λs = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
                (:debug, "Loading compatible initial conditions."),
                (:debug, "Initial conditions compat"),
                (:debug, "Stored values compat"),
                (:debug, "Incompatible values. Computing new values."),
                (:debug, "Selected compatible initial conditions"),
                (:debug, "Updating copy"),
                (:debug, "Updating in-place"),
                (:debug, "Deleting rows"),
                min_level=Logging.Debug, match_mode=:all,
                λmap(10., ic_alg=algs[i], alg=λa, recompute=true))
            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.3)
        end
    end
end

@testset "Other DB operations" begin
    λalgs = [DynSys(T=12000.)]
    algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
    counts = [2, 2, 4]
    for λa in λalgs
        for i ∈ eachindex(algs)
            λs = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
                (:debug, "Loading compatible initial conditions."),
                (:debug, "Initial conditions compat"),
                (:debug, "Stored values compat"),
                (:debug, "Incompatible values. Computing new values."),
                (:debug, "Selected compatible initial conditions"),
                (:debug, "Updating copy"),
                (:debug, "Cloning"),
                (:debug, "Deleting rows"),
                min_level=Logging.Debug, match_mode=:all,
                λmap(10., ic_alg=algs[i], alg=λa))
            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.3)
        end
    end
    db = InitialConditions.DataBase(params)
    @test size(db.df, 1) == (2+2+4)*2

    for λa in λalgs
        for i ∈ eachindex(algs)
            λs = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
                (:debug, "Incompatible initial conditions. Generating new conditions."),
                (:debug, "Generated $(counts[i]) initial conditions."),
                (:debug, "Deleting rows"),
                (:debug, "Appending with missing"),
                (:debug, "Initial conditions compat"),
                (:debug, "Stored values compat"),

                min_level=Logging.Debug, match_mode=:any,
                λmap(20., ic_alg=algs[i], alg=λa))
            @test length(λs) == counts[i]
            @test all(0 .< λs .< 0.4)
        end
    end

end

rm(TEST_DIR, force=true, recursive=true)

end
