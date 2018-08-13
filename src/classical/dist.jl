@everywhere begin
    using OrdinaryDiffEq
    using DiffEqMonteCarlo
    using DiffEqCallbacks
    using RecursiveArrayTools
    using StaticArrays
end

!contains(==, names(Main), :InitialConditions) && include("initial_conditions.jl")
include("parallel.jl")


using DataFrames, CSV
using InitialConditions

function monte_dist(f, p, u0::Array{SVector{N, T}}, d0, t, dt=0.01;
    parallel_type=:none,
    kwargs=Dict(:abstol=>1e-14, :reltol=>0, :maxiters=>1e9)) where {N, T}

    n = length(u0)
    tspan = (0., t)
    prob = dist_prob(f, p, u0[1], d0, tspan)

    prob_func(prob, i, repeat) = dist_prob(f, p, u0[i], d0, tspan)
    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = DiffEqArray([norm(s[idx1] - s[idx2]) for s in sol.u],sol.t)
        return d, false
    end
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    sim = solve(monte_prob, Vern9(); kwargs..., saveat=dt, num_monte=n, parallel_type=parallel_type)

    return sim
end

function monte_dist(p, p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, d0, t, dt=0.01;
    parallel_type=:none,
    kwargs=Dict(:abstol=>1e-14, :reltol=>0, :maxiters=>1e9)) where {N, T}

    n = length(q0)
    tspan = (0., t)
    prob = dist_prob(p, p0[1], q0[1], d0, tspan)

    prob_func(prob, i, repeat) = dist_prob(p, p0[i], q0[i], d0, tspan)
    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = DiffEqArray([norm(vcat(s[1,:][idx1] - s[1,:][idx2], s[2,:][idx1] - s[2,:][idx2])) for s in sol.u],sol.t)
        return d, false
    end
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    sim = solve(monte_prob, DPRKN12(); kwargs..., saveat=dt, num_monte=n, parallel_type=parallel_type)

    return sim
end

function d∞(p, p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, d0, t=500.;
    parallel_type=:none,
    kwargs=Dict(:abstol=>1e-14, :reltol=>0, :maxiters=>1e9)) where {N, T}

    n = length(q0)
    tspan = (0., t)
    prob = dist_prob(p, p0[1], q0[1], d0, tspan)

    prob_func(prob, i, repeat) = dist_prob(p, p0[i], q0[i], d0, tspan)
    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = norm(vcat(sol[1][1,:][idx1] - sol[1][1,:][idx2], sol[1][2,:][idx1] - sol[1][2,:][idx2]))
        return d, false
    end
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    sim = solve(monte_prob, DPRKN12(); kwargs..., save_start=false, save_everystep=false,
        num_monte=n, parallel_type=parallel_type)

    return mean(sim)
end

function d∞(E, p, d0, T=500.; parallel_type=:pmap, kwargs=Dict(:abstol=>1e-14, :reltol=>0, :maxiters=>1e9))
    q0, p0, n = generateInitialConditions(E, params=p)
    q0 = [SVector{2}(q0[i,1], q0[i,2]) for i=1:n]
    p0 = [SVector{2}(p0[i,1], p0[i,2]) for i=1:n]

    d∞(p, p0, q0, d0, T, parallel_type=parallel_type)
end
