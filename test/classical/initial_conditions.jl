using Test
using Logging
using DataFrames, CSV

@testset "Initial Conditions" begin

TEST_DIR = "output/classical/B0.55-D0.4/"

if isdir(TEST_DIR)
    rm(TEST_DIR, force=true, recursive=true)
end

params = PhysicalParameters()

@testset "No files" begin
    @test_logs((:debug, "No initial conditions file found. Generating new conditions."),
        (:debug, "Generated 2 initial conditions."),
        min_level=Logging.Debug, match_mode=:all,
        initial_conditions(10., alg=PoincareRand(n=2)))
end

@testset "Basic save / load" begin
    algs = [PoincareUniform(n=3,m=3), InscribedCircle(n=2, m=2)]
    counts = [2, 4]
    for i ∈ eachindex(algs)
        q0, p0 = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
            (:debug, "Incompatible initial conditions. Generating new conditions."),
            (:debug, "Generated $(counts[i]) initial conditions."),
            (:debug, "Appending with missing"),
            min_level=Logging.Debug, match_mode=:all,
            initial_conditions(10., alg=algs[i]))
        @test counts[i] == size(q0, 1)
        for i in axes(q0, 1)
            @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
        end
    end

    algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
    counts = [2, 2, 4]
    for i ∈ eachindex(algs)
        q0, p0 = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
            (:debug, "Loading compatible initial conditions."),
            min_level=Logging.Debug, match_mode=:all,
            initial_conditions(10., alg=algs[i]))
        @test counts[i] == size(q0, 1)
        for i in axes(q0, 1)
            @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
        end
    end
end

@testset "Recompute" begin
    algs = [PoincareRand(n=2), PoincareUniform(n=3, m=3), InscribedCircle(n=2, m=2)]
    counts = [2, 2, 4]
    for i ∈ eachindex(algs)
        q0, p0 = @test_logs((:debug, "Checking compatibility with stored initial conditions"),
            (:debug, "Incompatible initial conditions. Generating new conditions."),
            (:debug, "Generated $(counts[i]) initial conditions."),
            (:debug, "Deleting rows"),
            (:debug, "Appending with missing"),
            min_level=Logging.Debug, match_mode=:all,
            initial_conditions(10., alg=algs[i], recompute=true))
        @test counts[i] == size(q0, 1)
        for i in axes(q0, 1)
            @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
        end
    end
end

@testset "DB operations" begin
    E = 10.
    db = InitialConditions.DataBase(E, params)
    @test size(db.df, 1) == 2+2+4

    @testset "Insert after having other data" begin
        fake_data = 1:8
        db.df[:fake_data] = fake_data
        InitialConditions.DataBaseInterface.update_file(db)

        @test_logs((:debug, "Checking compatibility with stored initial conditions"),
            (:debug, "Incompatible initial conditions. Generating new conditions."),
            (:debug, "Appending with missing"),
            min_level=Logging.Debug, match_mode=:any,
            initial_conditions(10., alg=PoincareRand(n=3)))

        db = InitialConditions.DataBase(E, params)
        @test all(db.df[:fake_data][1:8] .== fake_data)
        @test all(db.df[:fake_data][9:end] .== nothing) # was missing, but converted at read

        @test_logs((:debug, "Checking compatibility with stored initial conditions"),
            (:debug, "Incompatible initial conditions. Generating new conditions."),
            (:debug, "Deleting rows"),
            min_level=Logging.Debug, match_mode=:any,
            initial_conditions(10., alg=PoincareRand(n=2), recompute=true))

        db = InitialConditions.DataBase(E, params)
        @test all(db.df[:fake_data][1:6] .== fake_data[3:8])
        @test all(db.df[:fake_data][9:end] .== nothing) # was missing, but converted at read
    end
end

rm(TEST_DIR, force=true, recursive=true)

end
