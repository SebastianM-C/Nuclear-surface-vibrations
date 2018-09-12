__precompile__(false)

module NuclearSurfaceVibrations

export Classical, Quantum

using Parameters
using Distributed

include("utils.jl")
using .Utils

@everywhere using Pkg
@everywhere Pkg.activate(".")

abstract type AbstractAlgorithm end

module Classical

using Reexport
using ..Utils
using ..Distributed
using ..Parameters
using ..NuclearSurfaceVibrations: AbstractAlgorithm

include("db.jl")
include("classical/hamiltonian.jl")
include("classical/parallel.jl")
include("classical/initial_conditions.jl")
@reexport using .InitialConditions
@reexport using .Hamiltonian
using .ParallelTrajectories

include("classical/poincare.jl")
include("classical/lyapunov.jl")
include("classical/dist.jl")

@reexport using .Poincare
@reexport using .Lyapunov
@reexport using .DInfty

include("classical/reductions.jl")

@reexport using .Reductions

end  # module Classical

module Quantum

using Reexport

# include("quantum/energylevels.jl")

# @reexport using .EnergyLevels

end  # module Quantum

end  # module NuclearSurfaceVibrations
