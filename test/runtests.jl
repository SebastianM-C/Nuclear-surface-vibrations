using Test
using NuclearSurfaceVibrations

include("db.jl")

@testset "Classical" begin
using .Classical
include("classical/initial_conditions.jl")
include("classical/lyapunov.jl")
# include("classical/dist.jl")
end

@testset "Quantum" begin
# include("quantum/hamiltonian.jl")
# include("quantum/regions.jl")
# include("quantum/irreducible_representations.jl")
end
