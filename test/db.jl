using Test
using StorageGraphs

include("../src/db.jl")
using .DataBaseInterface

@testset "DB loading" begin
    g = initialize()
    @test typeof(g) <: StorageGraph
end
