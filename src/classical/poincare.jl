#!/usr/bin/env julia

addprocs(Int(Sys.CPU_CORES / 2))

@everywhere begin
    using DiffEqBase, OrdinaryDiffEq, DiffEqMonteCarlo

    cb(idx, s) = DiffEqBase.ContinuousCallback((u, t, integrator)->s*u, 
        (integrator)->nothing, nothing,
        save_positions=(false, true), idxs=idx)

end

@everywhere include("hamiltonian.jl")
using .Hamiltonian

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
- `sgn = 1`: The intersection direction with the plane
"""
function poincaremap(E, q0, p0, N; A=1, B=0.55, D=0.4, t=100, axis=3, sgn=1)
    tspan = (0., t)
    prefix = "../../output/classical/B$B-D$D/E$E"
    if !isdir(prefix)
        mkpath(prefix)
    end

    z0 = hcat(p0, q0)

    # prob = HamiltonianProblem(H, q0[1,:], p0[1,:], tspan)
    # prob = DynamicalODEProblem(q̇, ṗ, q0[1,:], p0[1,:], tspan)
    prob = DiffEqBase.ODEProblem(ż, z0[1, :], tspan, (A, B, D), callback=cb(axis, sgn))

    function prob_func(prob, i, repeat)
        DiffEqBase.ODEProblem(prob.f, z0[i, :], prob.tspan, prob.p, callback=cb(axis, sgn))
    end

    monte_prob = DiffEqBase.MonteCarloProblem(prob, prob_func=prob_func)

    sim = DiffEqBase.solve(monte_prob, OrdinaryDiffEq.Vern9(), abstol=1e-14, reltol=1e-14,
        save_everystep=false, save_start=false, save_end=false,
        save_everystep=false, num_monte=N, parallel_type=:pmap)
end
