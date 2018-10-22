module ParallelTrajectories

export parallel_problem, create_parallel, parallel_evolution

using ..Distributed

using StaticArrays
using OrdinaryDiffEq
# using ChaosTools

function create_parallel(f::Function)
    @inbounds function eom(u::SVector{N}, p, t) where {N}
        idx1 = SVector{N÷2}(1:N÷2)
        idx2 = SVector{N÷2}(N÷2+1:N)
        vcat(f(u[idx1], p, t),
             f(u[idx2], p, t))
    end

    return eom
end

function create_parallel(f::Function, u)
    u₀ = vcat(u...)

    return create_parallel(f), u₀
end

function create_parallel(ṗ::Function, q̇::Function)
    @inbounds function f1(p::SVector{N}, q::SVector{N}, params, t) where {N}
        idx1 = SVector{N÷2}(1:N÷2)
        idx2 = SVector{N÷2}(N÷2+1:N)
        vcat(ṗ(p[idx1], q[idx1], params, t),
             ṗ(p[idx2], q[idx2], params, t))
    end

    @inbounds function f2(p::SVector{N}, q::SVector{N}, params, t) where {N}
        idx1 = SVector{N÷2}(1:N÷2)
        idx2 = SVector{N÷2}(N÷2+1:N)
        vcat(q̇(p[idx1], q[idx1], params, t),
             q̇(p[idx2], q[idx2], params, t))
    end

    return f1, f2
end

function create_parallel(ṗ::Function, q̇::Function, p0, q0)
    p₀ = vcat(p0...)
    q₀ = vcat(q0...)

    return create_parallel(ṗ, q̇)..., p₀, q₀
end

function parallel_problem(f, u, tspan, params, cb=nothing)
    peom, u₀ = create_parallel(f, u)
    ODEProblem(peom, u₀, tspan, params, callback=cb)
end

function parallel_problem(ṗ, q̇, p0, q0, tspan, params, cb=nothing)
    f1, f2, p₀, q₀ = create_parallel(ṗ, q̇, p0, q0)
    DynamicalODEProblem(f1, f2, p₀, q₀, tspan, params, callback=cb)
end

function parallel_evolution(ż, u0, t;
        params, parallel_type=:none, alg=Vern9(),
        output_func=(sol, i)->(sol, false), cb=nothing,
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9))

    n = length(u0)
    tspan = (zero(typeof(t)), t)
    f = create_parallel(ż)
    prob = ODEProblem(f, u0[1], tspan, params, callback=isa(cb, Vector) ? cb[1] : cb)

    prob_func(prob, i, repeat) = ODEProblem(f, u0[i], tspan, params,
        callback=isa(cb, Vector) ? cb[i] : cb)
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    sim = solve(monte_prob, alg; kwargs..., num_monte=n, parallel_type=parallel_type,
        save_start=save_start, save_everystep=save_everystep, saveat=saveat)

    return sim
end

function parallel_evolution(ż, z0::Array{SVector{N, T}}, d0, t;
        params, parallel_type=:none, alg=Vern9(),
        output_func=(sol, i)->(sol, false), cb=nothing,
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    u0 = [vcat(z, z.+d0/√N) for z ∈ z0]
    parallel_evolution(ż, u0, t, params=params, parallel_type=parallel_type,
        alg=alg, output_func=output_func, cb=cb, save_start=save_start,
        save_everystep=save_everystep, saveat=saveat, kwargs=kwargs)
end

function parallel_evolution(ṗ, q̇, u1, u2, tspan;
        params, parallel_type=:none, alg=DPRKN12(),
        output_func=(sol, i)->(sol, false), cb=nothing,
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9))

    @assert length(u1) == length(u2)
    n = length(u1)
    f1, f2 = create_parallel(ṗ, q̇)
    prob = DynamicalODEProblem(f1, f2, u1[1], u2[1], tspan, params,
        callback=isa(cb, Vector) ? cb[1] : cb)

    prob_func(prob, i, repeat) =
        DynamicalODEProblem(f1, f2, u1[i], u2[i], tspan, params,
            callback=isa(cb, Vector) ? cb[i] : cb)
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    sim = solve(monte_prob, alg; kwargs..., num_monte=n, parallel_type=parallel_type,
        save_start=save_start, save_everystep=save_everystep, saveat=saveat)

    return sim
end

function parallel_evolution(ṗ, q̇, p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}},
        d0, t; params, parallel_type=:none, alg=DPRKN12(),
        output_func=(sol, i)->(sol, false), cb=nothing,
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    u1 = [vcat(p, p.+d0/√(2N)) for p ∈ p0]
    u2 = [vcat(q, q.+d0/√(2N)) for q ∈ q0]

    parallel_evolution(ṗ, q̇, u1, u2, t, params=params, parallel_type=parallel_type,
        alg=alg, output_func=output_func, cb=cb, save_start=save_start,
        save_everystep=save_everystep, saveat=saveat, kwargs=kwargs)
end

# function parallel_problem(ds::ChaosTools.CDS, states, tspan)
#     peom, st = DynamicalSystemsBase.create_parallel(ds, states)
#     ODEProblem(peom, st, tspan, ds.p)
# end

end  # module ParallelTrajectories
