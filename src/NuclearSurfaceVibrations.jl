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
include("classical/custom.jl")
@reexport using .InitialConditions
@reexport using .Hamiltonian
using .ParallelTrajectories

include("classical/poincare.jl")
include("classical/dist.jl")
include("classical/lyapunov.jl")

@reexport using .Poincare
@reexport using .DInfty
@reexport using .Lyapunov

include("classical/reductions.jl")
include("classical/diagnostics.jl")

@reexport using .Reductions
@reexport using .Diagnostics

end  # module Classical

module Quantum

using Reexport

# include("quantum/energylevels.jl")

# @reexport using .EnergyLevels

end  # module Quantum

end  # module NuclearSurfaceVibrations
