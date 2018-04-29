#!/usr/bin/env julia
module Poincare

export poincare

addprocs(Int(Sys.CPU_CORES / 2))
using DiffBase, OrdinaryDiffEq, DiffEqMonteCarlo
using ParallelDataTransfer

include("initial_conditions.jl")
@everywhere include("hamiltonian.jl")
using .Hamiltonian, .InitialConditions

@everywhere begin
    condition(u, t, integrator) = u
    affect!(integrator) = nothing
    PoincareCb(idx) = DiffEqBase.ContinuousCallback(condition, affect!, nothing,
        save_positions=(false, true), idxs=idx)
end

"""
    poincaremap(E; n=10, m=10, A=1, B=0.55, D=0.4, t=100, axis=3)

Create a Poincare map at the given energy for the given parameters through
a Monte Carlo simulation.

## Arguments
- `E`: The energy of the system
- `n = 10`: Target number of initial conditions per kinetic energy values
- `m = 10`: Target number of kinetic energy values
- `A = 1`: Hamiltonian A parameter
- `B = 1`: Hamiltonian B parameter
- `D = 1`: Hamiltonian D parameter
- `t = 100`: Simulation duration
- `axis = 3`: Axis for the Poincare section
"""
function poincaremap(E; n=10, m=10, A=1, B=0.55, D=0.4, t=100, axis=3)
    tspan = (0., t)
    prefix = "../../output/classical/B$B-D$D/E$E"
    if !isdir(prefix)
        mkpath(prefix)
    end
    q0, p0, N = @time generateInitialConditions(E, n, m, params=(A, B, D))
    z0 = hcat(p0, q0)
    @everywhere cb = PoincareCb(axis)
    # prob = HamiltonianProblem(H, q0[1,:], p0[1,:], tspan)
    # prob = DynamicalODEProblem(q̇, ṗ, q0[1,:], p0[1,:], tspan)
    prob = ODEProblem(ż, z0[1, :], tspan, (A, B, D), callback=cb)

    sendto(workers(), prob=prob, z0=z0)

    @everywhere function prob_func(prob, i, repeat)
        DiffEqBase.ODEProblem(prob.f, z0[i, :], prob.tspan, prob.p, callback=cb)
    end

    monte_prob = MonteCarloProblem(prob, prob_func=prob_func)

    sim = solve(monte_prob, Vern9(), abstol=1e-14, reltol=1e-14,
        save_everystep=false, save_start=false, save_end=false,
        save_everystep=false, num_monte=N, parallel_type=:pmap)
end

end  # module Poincare
