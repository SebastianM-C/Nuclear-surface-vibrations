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
    # p₀ = prob.u0[2][1] + d0
    # p₂ = sum(prob.u0[2].^2) - p₀^2
    # if p₂ > 0
    #     p₂ = √(p₂)
    #     q0, p0 = prob.u0[1], [p₀, p₂]
    # elseif sum(prob.u0[2].^2) - (prob.u0[2][2] + d0)^2 > 0
    #     p₂ = prob.u0[2][2] + d0
    #     p₀ = √(sum(prob.u0[2].^2) - p₂^2)
    #     q0, p0 = prob.u0[1], [p₀, p₂]
    # else
    #     A, B, D = readdlm("param.dat")
    #     @inline V(q₀, q₂) = A / 2 * (q₀^2 + q₂^2) + B / √2 * q₀ * (3 * q₂^2 - q₀^2) + D / 4 * (q₀^2 + q₂^2)^2;
    #     @inline T(p₀, p₂) = A / 2 * (p₀^2 + p₂^2);
    #     @inline H(q₀, q₂, p₀, p₂) = T(p₀, p₂) + V(q₀, q₂);
    #     E = H(prob.u0[1][1], prob.u0[1][2], prob.u0[2][1], prob.u0[2][2])
    #     q₀ = prob.u0[1][1] + d0
    #     q₂ = Array{eltype(prob.u0[1])}(2)
    #     q₂[1] = (-(A / D) - (3 * √2 * B * q₀) / D - q₀^2 - √(A^2 + 4 * D * E +
    #       6 * √2 * A * B * q₀ + 18 * B^2 * q₀^2 + 8 * √2 * B * D * q₀^3) / D)
    #     q₂[2] = (-(A / D) - (3 * √2 * B * q₀) / D - q₀^2 + √(A^2 + 4 * D * E +
    #       6 * √2 * A * B * q₀ + 18 * B^2 * q₀^2 + 8 * √2 * B * D * q₀^3) / D)
    #     if q₂[1] > 0
    #         q0, p0 = [q₀, √(q₂[1])], prob.u0[2]
    #     elseif q₂[2] > 0
    #         q0, p0 = [q₀, √(q₂[2])], prob.u0[2]
    #     else
    #         println("Couldn't generate test trajectory with the same energy!")
    #         q0, p0 = prob.u0 .+ d0 / √2
    #         @show q0, p0
    #     end
    # end

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
