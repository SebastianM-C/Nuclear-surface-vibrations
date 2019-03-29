using Test
using StorageGraphs
using StorageGraphs: loadgraph

include("../src/db.jl")
using .DataBaseInterface


ic_deps = ((A=1,),(B=0.5,),(E=1,))
g[:a, (q=1,),(w=2,)]

@testset "DB creation" begin
    g = initalize()

end

end
