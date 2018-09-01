__precompile__(false)

module NuclearSurfaceVibrations

using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(".")

export Classical, Quantum

module Classical

export H, T, V, poincaremap, coloredpoincare

using Reexport
using Distributed

@everywhere include("$(@__DIR__)/classical/hamiltonian.jl")

include("$(@__DIR__)/classical/initial_conditions.jl")
# @everywhere include("$(@__DIR__)/classical/poincare.jl")

@everywhere using .Hamiltonian
@reexport using .InitialConditions

end  # module Classical

module Quantum

end  # module Quantum

end  # module NuclearSurfaceVibrations
