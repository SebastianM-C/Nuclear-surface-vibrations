using Test

@testset "Database Interface" begin
    include("db.jl")
    rm("test.csv")
end
