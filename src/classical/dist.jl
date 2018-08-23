using Distributed
@everywhere begin
    using OrdinaryDiffEq
    using DiffEqMonteCarlo
    using DiffEqCallbacks
    using RecursiveArrayTools
    using StaticArrays
end

!any(x->x==:InitialConditions, names(Main)) && @everywhere include("initial_conditions.jl")
include("parallel.jl")


using DataFrames, CSV
using .InitialConditions

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

    return sim
end

function d∞(E, p, d0=1e-8, T=500.; parallel_type=:pmap,
        kwargs=Dict(:abstol=>1e-14, :reltol=>0, :maxiters=>1e9),
        recompute=false)
    q0, p0, n = generateInitialConditions(E, params=p)
    q0 = [SVector{2}(q0[i,1], q0[i,2]) for i=1:n]
    p0 = [SVector{2}(p0[i,1], p0[i,2]) for i=1:n]

    A, B, D = p
    prefix = "../../output/classical/B$B-D$D/E$E"
    df = CSV.read("$prefix/z0.csv", allowmissing=:none, use_mmap=!is_windows())
    # Workaround for https://github.com/JuliaData/CSV.jl/issues/170
    if !haskey(df, :d∞) || recompute
        d = d∞(p, p0, q0, d0, T, parallel_type=parallel_type)
        df[:d∞] = d
        CSV.write("$prefix/z0.csv", df)
    else
        d = df[:d∞]
    end
    return d
end

function Γ(E, reduction, d0, p)
    λ(E) = reduction(λmap(E, B=p[2], d0=d0))
    d_inf(E) = mean(d∞(E, p, d0))
    Γ(λ(E), d_inf(E))
end

Γ(λ, d∞) = (exp(λ) - 1) / d∞
