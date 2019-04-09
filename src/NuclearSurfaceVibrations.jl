module NuclearSurfaceVibrations

export Classical, Quantum

using Parameters
using Reexport

# Fix for https://github.com/JuliaIO/ImageMagick.jl/issues/140
using ImageMagick

include("utils.jl")
include("db.jl")
using .Utils
using .DataBaseInterface

# Use the JULIA_PROJECT environment variable to pass the environment to the
# workers. See https://github.com/JuliaLang/julia/issues/28781
# @everywhere using Pkg
# @everywhere Pkg.activate(".")

abstract type AbstractAlgorithm end

module Classical

using Reexport
using Distributed
using Parameters
using ..Utils
using ..NuclearSurfaceVibrations: AbstractAlgorithm
using ..DataBaseInterface

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
include("classical/visualizations.jl")

@reexport using .Reductions
export Diagnostics
export Visualizations

end  # module Classical

module Quantum

using Reexport

# include("quantum/energylevels.jl")

# @reexport using .EnergyLevels

end  # module Quantum

end  # module NuclearSurfaceVibrations
