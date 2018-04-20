include("../../src/quantum/regions.jl")
using Regions
using Base.Test

v1 = [-0.32, 0.023, 0.2, 3.2, 3.2, 3.5]
v2 = [0.1, 2, 2.1, 3.45, 3.6]
v = (v1, v2)

@testset "Region by index tests" begin
    @test regions(v1, 2, byindex=true) == [[-0.32, 0.023, 0.2], [3.2, 3.2, 3.5]]
    @test regions(v2, 2, byindex=true) == [[0.1, 2], [2.1, 3.45, 3.6]]
    @test regions(v, 2, byindex=true) == ([[-0.32, 0.023, 0.2], [3.2, 3.2, 3.5]], [[0.1, 2], [2.1, 3.45, 3.6]])

    @test regions(v1, [0., 3.2]) == [[0.023, 0.2, 3.2]]
    @test regions(v2, [v2[1], (v2[end]-v2[1])/2, v2[end]]) == [[0.1], [2, 2.1, 3.45, 3.6]]
end

@testset "Region by energy tests" begin
    @test regions(v1, 2) == [[-0.32, 0.023, 0.2], [3.2, 3.2, 3.5]]
    @test regions(v2, 2) == regions(v2, [v2[1], (v2[end]-v2[1])/2, v2[end]])
    @test regions(v, 2) == ([[-0.32, 0.023, 0.2], [3.2, 3.2, 3.5]], [[0.1], [2, 2.1, 3.45, 3.6]])
    @test regions(v1, 10) == [[-0.32, 0.023], [0.2], [], [], [], [], [], [], [], [3.2, 3.2, 3.5]]
end
