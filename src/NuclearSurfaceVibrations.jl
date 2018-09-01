__precompile__(false)

module NuclearSurfaceVibrations

using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(".")

export Classical, Quantum

module Classical

export H, T, V, poincaremap, coloredpoincare

using Reexport
using ..Distributed

@everywhere include("$(@__DIR__)/classical/hamiltonian.jl")
include("classical/initial_conditions.jl")

@everywhere using .Hamiltonian
@reexport using .InitialConditions

@everywhere include("$(@__DIR__)/classical/poincare.jl")

end  # module Classical

module Quantum

using Reexport

# include("quantum/energylevels.jl")

# @reexport using .EnergyLevels

end  # module Quantum

end  # module NuclearSurfaceVibrations
