module Lyapunov

export λmap, λproblem, LyapunovAlgorithm, DynSys, TimeRescaling

using ..Distributed
using ..Parameters: @with_kw, @unpack
using ..Hamiltonian
using ..DataBaseInterface
using ..InitialConditions
using ..Classical: AbstractAlgorithm
using ..DInfty: monte_dist
using ..ParallelTrajectories

using ChaosTools
using Plots, LaTeXStrings
using OrdinaryDiffEq
using DiffEqCallbacks
using RecursiveArrayTools
using StaticArrays
using LightGraphs: outneighbors
using LinearAlgebra: norm

abstract type LyapunovAlgorithm <: AbstractAlgorithm end

@with_kw struct DynSys{R <: Real} <: LyapunovAlgorithm  @deftype R
    T = 1e4
    Ttr = 1e3
    d0 = 1e-9
    upper_threshold = 1e-6
    lower_threshold = 1e-12
    dt = 20.
    solver::OrdinaryDiffEqAlgorithm = Vern9()
    diff_eq_kwargs::NamedTuple = (abstol=1e-14, reltol=1e-14, maxiters=1e9)
end

@with_kw struct TimeRescaling{R <: Real} <: LyapunovAlgorithm  @deftype R
    T = 1e4
    Ttr = 1e3
    τ = 5.
    d0 = 1e-9
    solver::OrdinaryDiffEqAlgorithm = Vern9()
    diff_eq_kwargs::NamedTuple = (abstol=1e-14, reltol=1e-14, maxiters=1e9)
end

function λmap(E; params=PhysicalParameters(), ic_alg=PoincareRand(n=500),
        ic_recompute=false, alg=DynSys(), recompute=false)

    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    g = initalize()
    q0, p0 = initial_conditions(E, alg=ic_alg, params=params, recompute=ic_recompute)

    ic_deps = InitialConditions.depchain(params, E, ic_alg)
    pahs = paths_through(g, foldr(=>, ic_deps))
    if mapreduce(v->has_prop(g, v, :λ_alg), &, outneighbors(g, g[ic_deps[end]]))
        λs = g[:λ, ic_deps..., (λ_alg=alg,)]
    else
        λs = λmap(p0, q0, alg; params=params)
        add_derived_values!(g, ic_deps, (q₀=q0[:,1],q₂=q0[:,2], p₀=p0[:,1],p₂=p0[:,2]), (λ=λs,), (λ_alg=alg,))
        savechanges(g)
    end

    return λs
    # # FIXME
    # plt = histogram(disallowmissing(Array{arr_type}(λs)), nbins=50, xlabel=L"\lambda", ylabel=L"N", label="T = $T")
    # fn = string(typeof(alg)) * "_T$T" * "_hist"
    # fn = replace(fn, "{Float64}" => "")
    # savefig(plt, "$prefix/"*string(typeof(ic_alg))*"/lyapunov_$fn.pdf")
    # return disallowmissing(Array{arr_type}(λs))
end

# Too slow
#
# function λmap(q0::Array{SVector{N, X}}, p0::Array{SVector{N, X}}, alg::TimeRescaling,
#         timeseries::Val=Val(false); params=PhysicalParameters(), parallel_type=:none) where {N, X}
#     @unpack T, Ttr, d0, τ, solver, diff_eq_kwargs = alg
#
#     idx1 = SVector{N}(1:N)
#     idx2 = SVector{N}(N+1:2N)
#     @assert length(p0) == length(q0)
#     n = length(p0)
#
#     @inbounds dist(u) =
#         norm(vcat(u[1,:][idx1] - u[1,:][idx2], u[2,:][idx1] - u[2,:][idx2]))
#
#     function affect!(integrator)
#         a = dist(integrator.u)/d0
#         Δp = integrator.u[1,:][idx1] - integrator.u[1,:][idx2]
#         Δq = integrator.u[2,:][idx1] - integrator.u[2,:][idx2]
#         integrator.u = ArrayPartition(vcat(integrator.u.x[1][idx1],
#                                             integrator.u.x[1][idx1] + Δp/a),
#                                       vcat(integrator.u.x[2][idx1],
#                                             integrator.u.x[2][idx1] + Δq/a))
#     end
#     rescale = PeriodicCallback(affect!, τ, save_positions=(true, false))
#
#     function output_func_full(sol, i)
#         λᵢ = [log(dist(s)/d0)/τ for s in sol.u]
#         n = length(sol.t)
#         λ_series = DiffEqArray([sum(λᵢ[1:i]) for i in eachindex(λᵢ)]./(1:n), sol.t)
#         return λ_series, false
#     end
#
#     function output_func_end(sol, i)
#         λ = sum(log.(dist.(sol.u ./ d0))) / (length(sol) * τ)
#         return λ, false
#     end
#
#     output_func = timeseries == Val(false) ? output_func_end : output_func_full
#
#     if Ttr ≠ 0
#         sim_tr = parallel_evolution(ṗ, q̇, p0, q0, d0, Ttr, params=params, parallel_type=parallel_type,
#             save_start=false, save_everystep=false, alg=solver, kwargs=diff_eq_kwargs)
#         p0 = [sim_tr.u[i].u[end][1,:] for i=1:n]
#         q0 = [sim_tr.u[i].u[end][2,:] for i=1:n]
#
#         parallel_evolution(ṗ, q̇, p0, q0, T, params=params, parallel_type=parallel_type,
#             alg=solver, saveat=τ, output_func=output_func, cb=rescale, kwargs=diff_eq_kwargs)
#     else
#         parallel_evolution(ṗ, q̇, p0, q0, d0, T, params=params, parallel_type=parallel_type,
#             alg=solver, saveat=τ, output_func=output_func, cb=rescale, kwargs=diff_eq_kwargs)
#     end
# end

function λproblem(p0::SVector{N}, q0::SVector{N}, alg::TimeRescaling;
        params=PhysicalParameters()) where {N}
    @unpack T, Ttr, d0, τ, solver, diff_eq_kwargs = alg

    idx1 = SVector{N}(1:N)
    idx2 = SVector{N}(N+1:2N)
    @assert length(p0) == length(q0)

    @inbounds dist(u) =
        norm(vcat(u[1,:][idx1] - u[1,:][idx2], u[2,:][idx1] - u[2,:][idx2]))

    function affect!(integrator)
        a = dist(integrator.u)/d0
        Δp = integrator.u[1,:][idx1] - integrator.u[1,:][idx2]
        Δq = integrator.u[2,:][idx1] - integrator.u[2,:][idx2]
        integrator.u = ArrayPartition(vcat(integrator.u.x[1][idx1],
                                            integrator.u.x[1][idx1] + Δp/a),
                                      vcat(integrator.u.x[2][idx1],
                                            integrator.u.x[2][idx1] + Δq/a))
    end
    rescale = PeriodicCallback(affect!, τ, save_positions=(false, false))
    a = SavedValues(typeof(T), eltype(q0[1]))
    save_func(u, t, integrator) = log(dist(u) / d0)
    sc = SavingCallback(save_func, a, saveat=zero(T):τ:T)
    cs = CallbackSet(sc, rescale)

    if Ttr ≠ 0
        prob_tr = parallel_problem(ṗ, q̇, [p0, p0.+d0/√(2N)], [q0, q0.+d0/√(2N)], Ttr, params)
        sol_tr = solve(prob_tr, solver; diff_eq_kwargs..., save_start=false, save_everystep=false)

        remake(prob_tr, tspan=T, callback=cs, u0=sol_tr[end])
    else
        parallel_problem(ṗ, q̇, [p0, p0.+d0/√(2N)], [q0, q0.+d0/√(2N)], T, params, cs)
    end
end

function λproblem(z0::SVector{N}, alg::TimeRescaling;
        params=PhysicalParameters()) where {N}
    @unpack T, Ttr, d0, τ, solver, diff_eq_kwargs = alg

    idx1 = SVector{N}(1:N)
    idx2 = SVector{N}(N+1:2N)

    @inbounds dist(u) = norm(u[idx1] - u[idx2])

    function affect!(integrator)
        a = dist(integrator.u)/d0
        Δu = integrator.u[idx1] - integrator.u[idx2]
        integrator.u = vcat(integrator.u[idx1], integrator.u[idx1] + Δu/a)
    end
    rescale = PeriodicCallback(affect!, τ, save_positions=(false, false))
    a = SavedValues(typeof(T), eltype(z0[1]))
    save_func(u, t, integrator) = log(dist(u) / d0)
    sc = SavingCallback(save_func, a, saveat=zero(T):τ:T)
    cs = CallbackSet(sc, rescale)

    if Ttr ≠ 0
        prob_tr = parallel_problem(ż, [z0, z0.+d0/√N], Ttr, params)
        sol_tr = solve(prob_tr, solver; diff_eq_kwargs..., save_start=false, save_everystep=false)

        remake(prob_tr, tspan=T, callback=cs, u0=sol_tr[end])
    else
        parallel_problem(ż, [z0, z0.+d0/√N], T, params, cs)
    end
end

function λ(sol)
    a = sol.prob.callback.discrete_callbacks[1].affect!.saved_values
    τ = a.t[2] - a.t[1]
    DiffEqArray([sum(a.saveval[1:i]) / (τ*i) for i in eachindex(a.saveval)], a.t)
end

function λmap(p0::Array{SVector{N, X}}, q0::Array{SVector{N, X}}, alg::TimeRescaling;
        params=PhysicalParameters(), parallel_type=:none) where {N, X}
    λprob = λproblem(p0[1], q0[1], alg)
    prob_func(prob, i, repeat) = λproblem(p0[i], q0[i], alg)
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func)
    sim = solve(monte_prob, solver; diff_eq_kwargs..., num_monte=length(p0),
        save_start=false, save_everystep=false, parallel_type=parallel_type)

    λs = [λ(sim[i]) for i ∈ eachindex(sim)]
end

function λmap(z0::SVector{N, X}, alg::TimeRescaling;
        params=PhysicalParameters(), parallel_type=:none) where {N, X}
    λprob = λproblem(z0[1], alg)
    prob_func(prob, i, repeat) = λproblem(z0[i], alg)
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func)
    sim = solve(monte_prob, solver; diff_eq_kwargs..., num_monte=length(z0),
        save_start=false, save_everystep=false, parallel_type=parallel_type)

    λs = [λ(sim[i]) for i ∈ eachindex(sim)]
end

function λmap(p0, q0, alg::TimeRescaling; params::PhysicalParameters)
    p0 = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
    q0 = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
    if issplit(alg.solver)
        λmap(p0, q0, alg, params=params, parallel_type=:pmap)
    else
        z0 = [vcat(p0[i], q0[i]) for i ∈ axes(q0, 1)]
        λmap(z0, alg, params=params, parallel_type=:pmap)
    end
end

function λmap(p0, q0, alg::DynSys; params::PhysicalParameters)
    z0 = [SVector{4}(vcat(p0[i, :], q0[i, :])) for i ∈ axes(q0, 1)]
    λmap(z0, alg, params=params)
end

function λmap(z0, alg::DynSys; params=PhysicalParameters())
    @unpack T, Ttr, d0, upper_threshold, lower_threshold, dt, solver, diff_eq_kwargs = alg
    inittest = ChaosTools.inittest_default(4)
    ds = ContinuousDynamicalSystem(ż, z0[1], params)

    pinteg = DynamicalSystemsBase.parallel_integrator(ds,
            [deepcopy(DynamicalSystemsBase.get_state(ds)),
            inittest(DynamicalSystemsBase.get_state(ds), d0)]; alg=solver,
            diff_eq_kwargs...)

    λs = pmap(eachindex(z0)) do i
            reinit!(pinteg, [z0[i], inittest(z0[i], d0)])
            lyapunov(pinteg, T, Ttr, dt, d0, upper_threshold, lower_threshold)
        end

    return λs
end



function λ_time(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, d0=1e-9, t=1e4, ttr=100., τ=5.;
    parallel_type=:none,
    kwargs=Dict(:abstol=>1e-14, :reltol=>0, :maxiters=>1e9)) where {N, T}

    n = length(p0)
    f1, f2, p₀, q₀ = create_parallel(p, p0[1], q0[1], d0)

    if ttr ≠ 0
        tspan = (0., ttr)
        prob = DynamicalODEProblem(f1, f2, p₀, q₀, tspan, p)

        prob_func_tr(prob, i, repeat) = dist_prob(p, p0[i], q0[i], d0, tspan)
        monte_prob = MonteCarloProblem(prob, prob_func=prob_func_tr)

        sim_tr = solve(monte_prob, DPRKN12(); kwargs..., save_everystep=false,
            num_monte=n, parallel_type=parallel_type)

        p0 = [sim_tr.u[i].u[end][1,:] for i=1:n]
        q0 = [sim_tr.u[i].u[end][2,:] for i=1:n]
        p₀ = p0[1]
        q₀ = q0[1]
    end

    tspan = (ttr, ttr + t)
    prob = DynamicalODEProblem(f1, f2, p₀, q₀, tspan, p)
    idx1 = SVector{N}(1:N)
    idx2 = SVector{N}(N+1:2N)

    dist(u) = norm(vcat(u[1,:][idx1] - u[1,:][idx2], u[2,:][idx1] - u[2,:][idx2]))

    function affect!(integrator)
        a = dist(integrator.u)/d0
        Δp = integrator.u[1,:][idx1] - integrator.u[1,:][idx2]
        Δq = integrator.u[2,:][idx1] - integrator.u[2,:][idx2]
        integrator.u = ArrayPartition(vcat(integrator.u.x[1][idx1],
                                            integrator.u.x[1][idx1] + Δp/a),
                                      vcat(integrator.u.x[2][idx1],
                                            integrator.u.x[2][idx1] + Δq/a))
    end
    rescale = PeriodicCallback(affect!, τ, save_positions=(true,false))
    prob_func(prob, i, repeat) = ttr≠0 ?
        DynamicalODEProblem(f1, f2, p0[i], q0[i], tspan, p, callback=rescale) :
        dist_prob(p, p0[i], q0[i], d0, tspan, rescale)

    function output_func(sol, i)
        λᵢ = [log(dist(s)/d0)/τ for s in sol.u]
        n = length(sol.t)
        λ_series = DiffEqArray([sum(λᵢ[1:i]) for i in eachindex(λᵢ)]./(1:n), sol.t)
        return λ_series[end], false
    end
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    solve(monte_prob, DPRKN12(); kwargs..., saveat=τ, num_monte=n, parallel_type=parallel_type)
end

end  # module Lyapunov
