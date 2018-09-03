__precompile__(false)

module NuclearSurfaceVibrations

export Classical, Quantum

using Parameters
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(".")

abstract type AbstractAlgorithm end

module Classical

using Reexport
using ..Distributed
using ..Parameters
using ..NuclearSurfaceVibrations: AbstractAlgorithm

include("classical/hamiltonian.jl")
include("classical/initial_conditions.jl")
@reexport using .InitialConditions
@reexport using .Hamiltonian

include("classical/poincare.jl")
include("classical/lyapunov.jl")

@reexport using .Poincare
@reexport using .Lyapunov

end  # module Classical

module Quantum

using Reexport

# include("quantum/energylevels.jl")

# @reexport using .EnergyLevels

end  # module Quantum

end  # module NuclearSurfaceVibrations
