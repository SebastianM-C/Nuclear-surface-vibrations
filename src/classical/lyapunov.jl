#!/usr/bin/env julia
module Lyapunov
include("problem.jl")

using OrdinaryDiffEq, DynamicalSystems

export compute_lyapunov, lyapunov_timeseries

function initialize(prob::ODEProblem, d0, threshold, diff_eq_kwargs)

    threshold <= d0 && throw(ArgumentError("Threshold must be bigger than d0!"))

    if haskey(diff_eq_kwargs, :solver)
        solver = diff_eq_kwargs[:solver]
    else
        println("Using DPRKN12 as default solver")
        solver = DPRKN12()
    end
    # Initialize
    integ1 = init(prob, solver; diff_eq_kwargs..., save_first=false, save_everystep=false)
    integ1.opts.advance_to_tstop = true

    q0, p0 = prob.u0 .+ d0 / √2

    prob2 = HamiltonEqs.defineProblem(q0, p0, prob.tspan)
    integ2 = init(prob2, solver; diff_eq_kwargs..., save_first=false, save_everystep=false)
    integ2.opts.advance_to_tstop = true

    return integ1, integ2
end

function compute_lyapunov(prob::ODEProblem; d0=1e-9, threshold=10^4*d0, dt = 0.1,
    diff_eq_kwargs = Dict(:abstol=>d0, :reltol=>d0))

    integ1, integ2 = initialize(prob, d0, threshold, diff_eq_kwargs)
    DynamicalSystems.lyapunov_final(integ1, integ2, prob.tspan[2],
        d0, threshold, dt)
end

function lyapunov_timeseries(prob::ODEProblem; d0=1e-9, threshold=10^4*d0, dt = 0.1,
    diff_eq_kwargs = Dict(:abstol=>d0, :reltol=>d0))

    integ1, integ2 = initialize(prob, d0, threshold, diff_eq_kwargs)
    DynamicalSystems.lyapunov_full(integ1, integ2, prob.tspan[2],
        d0, threshold, dt)
end

end
