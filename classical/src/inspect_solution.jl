#!/usr/bin/env julia
using OrdinaryDiffEq, DiffEqCallbacks
using Plots
plotly()

function inspect_solution(prob::ODEProblem, axis)
    full_sol = solve(prob, KahanLi8(), dt=1e-2)
    if axis == 1        # q₀ == 0
        # Plot the full solution
        plt = plot(full_sol, vars=(1,2,4), ms=2.5, xaxis="q₀", yaxis="q₂", zaxis="p₂")
        m = minimum(full_sol[2,:])
        M = maximum(full_sol[2,:])
        m2 = minimum(full_sol[4,:])
        M2 = maximum(full_sol[4,:])
        yrange = m:M-m:M
        # Plot intersection plane
        surface!(zeros(2), yrange, [1.2m2 1.2M2; 1.2m2 1.2M2], α=0.4, legend=false)
        # Find the intersection with the solution
        sol = solve(prob, KahanLi8(), save_everystep = false, callback=cb1,
            save_start=false, dt=1e-2)
        # Plot the points on the intersection plane
        scatter3d!(sol, vars=(1,2,4), ms=1,legend=false)
    elseif axis == 2    # q₂ == 0
        # Plot the full solution
        plt = plot(full_sol, vars=(1,2,3), ms=2.5, xaxis="q₀", yaxis="q₂", zaxis="p₀")
        m = minimum(full_sol[1,:])
        M = maximum(full_sol[1,:])
        m2 = minimum(full_sol[3,:])
        M2 = maximum(full_sol[3,:])
        xrange = m:M-m:M
        # Plot intersection plane
        surface!(xrange, zeros(2), [1.2m2 1.2m2; 1.2M2 1.2M2; 0 0], α=0.4)
        # Find the intersection with the solution
        sol = solve(prob, KahanLi8(), save_everystep = false, callback=cb2,
            save_start=false, dt=1e-2)
        # Plot the points on the intersection plane
        scatter3d!(sol, vars=(1,2,3), ms=1,legend=false)
    end
    plt
end

# plotlyjs()
# inspect_solution(prob, 1)

# plot(sol.t, map(u->H(u...), sol.u) - E)

# Debug
