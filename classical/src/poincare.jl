#!/usr/bin/env julia
module Poincare
using OrdinaryDiffEq, DiffEqCallbacks
using Plots, LaTeXStrings
using Juno
using ProgressMeter

include("problem.jl")

export createMap, createMap!

# Construct ContiunousCallbacks for the Poincare section
condition1(t, u, integrator) = u[1]           # q₀ == 0
condition2(t, u, integrator) = u[2]           # q₂ == 0
affect!(integrator) = nothing
cb1 = ContinuousCallback(condition1, affect!, affect!,
                        save_positions = (false, true));
cb2 = ContinuousCallback(condition2, affect!, affect!,
                        save_positions = (false, true));

function poincareSection!(prob::ODEProblem, alg, plt::AbstractPlot, axis::Int,
        sgn::Int; kwargs=Dict(), plot_kwargs=Dict(:markersize=>0.5))
    sol = solve(prob, alg; kwargs..., save_everystep=false, save_start=true, maxiters=1e10)
    if axis == 1        # q₀ == 0   &   p₀ >= 0
        filter = sgn .* sol[3,:][2:end-1] .> 0
        x = sol[2,:][2:end-1][filter]
        y = sol[4,:][2:end-1][filter]
        plane = "q₀ == 0 plane"
    else                # q₂ == 0   &   p₂ >= 0
        filter = sgn .* sol[4,:][2:end-1] .> 0
        x = sol[1,:][2:end-1][filter]
        y = sol[3,:][2:end-1][filter]
        plane = "q₂ == 0 plane"
    end
    if size(sol[1,:], 1) > 2
        scatter!(plt, x, y, markerstrokecolor=nothing, legend=false; plot_kwargs...);
    else
        warn("No intersection with $plane for u0=$(prob.u0)")
    end
    nothing
end

function poincareSection!(prob::ODEProblem, λ, alg, plt::AbstractPlot, axis::Int,
        sgn::Int; kwargs=Dict(), plot_kwargs=Dict(:markersize=>0.5))
    sol = solve(prob, alg; kwargs..., save_everystep=false, save_start=true, maxiters=1e10)
    if axis == 1        # q₀ == 0   &   p₀ >= 0
        filter = sgn .* sol[3,:][2:end-1] .> 0
        x = sol[2,:][2:end-1][filter]
        y = sol[4,:][2:end-1][filter]
        plane = "q₀ == 0 plane"
    else                # q₂ == 0   &   p₂ >= 0
        filter = sgn .* sol[4,:][2:end-1] .> 0
        x = sol[1,:][2:end-1][filter]
        y = sol[3,:][2:end-1][filter]
        plane = "q₂ == 0 plane"
    end
    if size(sol[1,:], 1) > 2
        scatter!(plt, x, y, zcolor=fill(λ, size(x, 1)), colorbar_title=L"\lambda",
                markerstrokecolor=nothing, legend=false; plot_kwargs...);
    else
        warn("No intersection with $plane for u0=$(prob.u0)")
    end
    nothing
end

function initialise(solver, axis, H, E)
    if solver == "Vern9+ManifoldProjection"
        # Manifold Projection for energy conservation
        function g(u, resid)
            resid[1] = H(u[1], u[2], u[3], u[4]) - E
            resid[2:4] .= 0
        end
        mp = ManifoldProjection(g, save=false)
        if axis == 1        # q₀ == 0
            cb = CallbackSet(mp, cb1)
        elseif axis == 2    # q₂ == 0
            cb = CallbackSet(mp, cb2)
        end
        kwargs = Dict(:abstol=>1e-14, :reltol=>1e-14, :callback=>cb)
        alg = Vern9()
    elseif solver == "KahanLi8"
        if axis == 1        # q₀ == 0
            cb = cb1
        elseif axis == 2    # q₂ == 0
            cb = cb2
        end
        kwargs = Dict(:dt=>1e-2, :callback=>cb)
        alg = KahanLi8()
    else
        error("No configuration for $solver")
    end

    alg, kwargs
end

function createMap(defProb, q0list, p0list, tspan, solver, axis, sgn, H, E;
        progress_info="Progress: ", plot_kwargs=Dict(:markersize=>0.5))
    @assert size(q0list, 1) == size(p0list, 1) "q0 and p0 have different sizes!"
    if axis == 1        # q₀ == 0
        plt = plot(xaxis=L"$q_2$", yaxis=L"$p_2$");
    elseif axis == 2    # q₂ == 0
        plt = plot(xaxis=L"$q_0$", yaxis=L"$p_0$");
    else
        error("Invalid axis value $axis !")
    end
    createMap!(plt, defProb, q0list, p0list, tspan, solver, axis, sgn, H, E;
        progress_info=progress_info, plot_kwargs=plot_kwargs)
    return plt
end

function createMap!(plt, defProb, q0list, p0list, tspan, solver, axis, sgn, H, E;
        progress_info="Progress: ", plot_kwargs=Dict(:markersize=>0.5))
    @assert size(q0list, 1) == size(p0list, 1) "q0 and p0 have different sizes!"
    alg, kwargs = initialise(solver, axis, H, E)
    @showprogress progress_info for i in 1:size(q0list, 1)
    # @progress for i in 1:size(q0list, 1)
        prob = defProb(q0list[i,:], p0list[i,:], tspan)
        poincareSection!(prob, alg, plt, axis, sgn, kwargs=kwargs,
            plot_kwargs=plot_kwargs)
    end
end

function createMap(defProb, q0list, p0list, tspan, λs, solver, axis, sgn, H, E;
        progress_info="Progress: ", plot_kwargs=Dict(:markersize=>0.5))
    @assert size(q0list, 1) == size(p0list, 1) "q0 and p0 have different sizes!"
    @assert size(q0list, 1) == size(λs, 1) "q0 and λs have different sizes!"
    if axis == 1        # q₀ == 0
        plt = plot(xaxis=L"$q_2$", yaxis=L"$p_2$");
    elseif axis == 2    # q₂ == 0
        plt = plot(xaxis=L"$q_0$", yaxis=L"$p_0$");
    else
        error("Invalid axis value $axis !")
    end
    createMap!(plt, defProb, q0list, p0list, tspan, λs, solver, axis, sgn, H, E;
        progress_info=progress_info, plot_kwargs=plot_kwargs)
    return plt
end

function createMap!(plt, defProb, q0list, p0list, tspan, λs, solver, axis, sgn, H, E;
        progress_info="Progress: ", plot_kwargs=Dict(:markersize=>0.5))
    @assert size(q0list, 1) == size(p0list, 1) "q0 and p0 have different sizes!"
    @assert size(q0list, 1) == size(λs, 1) "q0 and λs have different sizes!"
    alg, kwargs = initialise(solver, axis, H, E)
    @showprogress progress_info for i in 1:size(q0list, 1)
    # @progress for i in 1:size(q0list, 1)
        prob = defProb(q0list[i,:], p0list[i,:], tspan)
        poincareSection!(prob, λs[i], alg, plt, axis, sgn, kwargs=kwargs,
            plot_kwargs=plot_kwargs)
    end
end

end # module Poincare
