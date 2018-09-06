include("../../src/quantum/hamiltonian.jl")
using .Hamiltonian
using SparseArrays
using Test

@testset "Hamiltonian computation tests" begin
    H = [ 0.0  0.0  0.0  0.0  0.0  0.0
     0.0  1.0  0.0  0.0  0.0  0.0
     0.0  0.0  2.0  0.0  0.0  0.0
     0.0  0.0  0.0  1.0  0.0  0.0
     0.0  0.0  0.0  0.0  2.0  0.0
     0.0  0.0  0.0  0.0  0.0  2.0]
    H_ = Hamiltonian.compute_dense_hamiltonian(3, b=0, d=0)
    @test H_ ≈ H
    @test compute_hamiltonian(3, b=0, d=0) ≈ sparse(H)

    H = [0.0 0.0 0.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0 0.825 0.0
    0.0 0.0 2.3 0.5833630944789018 0.0 0.1
    0.0 0.0 0.5833630944789018 1.0 0.0 -0.5833630944789018
    0.0 0.825 0.0 0.0 2.2 0.0
    0.0 0.0 0.1 -0.5833630944789018 0.0 2.3]

    @test compute_hamiltonian(3) ≈ sparse(H)
end
