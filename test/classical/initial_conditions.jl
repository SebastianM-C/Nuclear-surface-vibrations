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
        min_level=Logging.Debug, match_mode=:any,
        initial_conditions(10., alg=PoincareRand(n=2)))
end

@testset "Basic save / load" begin
    @test_logs((:debug, "Loading compatible initial conditions."),
        min_level=Logging.Debug, match_mode=:any,
        initial_conditions(10., alg=PoincareRand(n=2)))
    q0, p0 = initial_conditions(10., alg=PoincareRand(n=2), params=params)
    for i in axes(q0, 1)
        @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
    end

    for alg in [PoincareUniform(n=3,m=3), InscribedCircle(n=3,m=3)]
        @test_logs((:debug, "Incompatible initial conditions. Generating new conditions."),
            min_level=Logging.Debug, match_mode=:any,
            initial_conditions(10., alg=alg))
        q0, p0 = initial_conditions(10., alg=alg, params=params)
        for i in axes(q0, 1)
            @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
        end
    end

    for alg in [PoincareRand(n=2), PoincareUniform(n=3,m=3), InscribedCircle(n=3,m=3)]
        @test_logs((:debug, "Loading compatible initial conditions."),
            min_level=Logging.Debug, match_mode=:any,
            initial_conditions(10., alg=alg))
        q0, p0 = initial_conditions(10., alg=alg, params=params)
        for i in axes(q0, 1)
            @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
        end
    end
end

@testset "Recompute" begin
    for alg in [PoincareRand(n=2), PoincareUniform(n=3,m=3), InscribedCircle(n=3,m=3)]
        @test_logs((:debug, "Incompatible initial conditions. Generating new conditions."),
            min_level=Logging.Debug, match_mode=:any,
            initial_conditions(10., alg=alg, recompute=true))
        q0, p0 = initial_conditions(10., alg=alg, params=params)
        for i in axes(q0, 1)
            @test H(p0[i,:], q0[i,:], params) - 10. ≈ 0 atol=1e-12
        end
    end
end

@testset "DB operations" begin
    E = 10.
    db = InitialConditions.DataBase(E, params)
    @test size(db.df, 1) == 2+2+9

    fake_data = rand(13)
    db.df[:fake_data] = fake_data
    InitialConditions.DataBaseInterface.update_file(db)

    @test_logs((:debug, "Incompatible initial conditions. Generating new conditions."),
        (:debug, "Appending"),
        min_level=Logging.Debug, match_mode=:any,
        initial_conditions(10., alg=PoincareRand(n=3)))

    db = InitialConditions.DataBase(E, params)
    @test all(db.df[:fake_data][1:13] .== fake_data)
    @test all(db.df[:fake_data][13:3] .=== missing)
end

rm(TEST_DIR, force=true, recursive=true)

end
