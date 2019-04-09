using Test
using StorageGraphs

@testset "DB loading" begin
    g = Classical.initialize()
    @test typeof(g) <: StorageGraph
end
