module Lyapunov

export λproblem, λmap, λ_timeseries_problem, λ_timeseries_map, LyapunovAlgorithm,
    DynSys, TimeRescaling

using ..Distributed
using ..Parameters: @with_kw, @unpack
using ..Hamiltonian
using ..DataBaseInterface
using ..InitialConditions
using ..InitialConditions: depchain, initial_conditions!
using ..Classical: AbstractAlgorithm
using ..DInfty: monte_dist
using ..ParallelTrajectories
using ..Custom

using ChaosTools
using Plots, LaTeXStrings
using OrdinaryDiffEq
using DiffEqCallbacks
using RecursiveArrayTools
using StaticArrays
using LightGraphs: outneighbors
using LinearAlgebra: norm
using StorageGraphs

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

function λhist(λs, params, alg, E, ic_alg)
    T = alg.T
    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    plt = histogram(λs, nbins=50, xlabel=L"\lambda", ylabel=L"N", label="T = $T")
    fn = string(typeof(alg)) * "_T$T" * "_hist"
    fn = replace(fn, "{Float64}" => "")
    fn = replace(fn, "NuclearSurfaceVibrations.Classical.Lyapunov." => "")
    dir = "$prefix/"*string(typeof(ic_alg))
    dir = replace(dir, "NuclearSurfaceVibrations.Classical.InitialConditions." => "")
    if !isdir(dir)
        mkpath(dir)
    end
    @debug "Saving plot"
    savefig(plt, dir*"/lyapunov_$fn.pdf")
end

function computeλ(node, alg, ic_alg, ic_dep, params, E, g)
    q0 = node[:q0]
    p0 = node[:p0]
    λs, t = @timed λmap(p0, q0, alg; params=params)
    @debug "Computing values took $t seconds."
    _, t = @timed add_nodes!(g, foldr(=>,(ic_dep..., node, (λ_alg=alg,), (λ=λs,))))
    @debug "Adding λs to graph took $t seconds."

    return λs
end

function λmap!(g::StorageGraph, E; params=PhysicalParameters(), ic_alg=PoincareRand(n=500),
        ic_recompute=false, alg=DynSys(), recompute=false)
    ic_node = initial_conditions!(g, E, alg=ic_alg, params=params, recompute=ic_recompute)
    ic_dep = depchain(params, E, ic_alg)
    # ic_node is the node with the compatible initial conditions
    # we need to check if λs were computed for this node
    outn = outneighbors(g, ic_node)

    if length(outn) > 0 && any(has_prop.(Ref(g), outn, :λ_alg))
        # we have some λs computed, we now have to check if they are the
        # right ones
        @debug "Looking for available λ_algs"
        idx = findfirst(v->g[v] == (λ_alg=alg,), outn)
        if idx !== nothing
            @debug "Found compatible λs"
            vals, t = @timed g[:λ, ic_dep..., (λ_alg=alg,)]
            @assert length(vals) == 1 "Lyapunov values not uniquely represented by deps!"
            λs = vals[1]
            @debug "Loading saved λs took $t seconds."
        else
            λs = computeλ(ic_node, alg, ic_alg, ic_dep, params, E, g)
        end
    else
        λs = computeλ(ic_node, alg, ic_alg, ic_dep, params, E, g)
    end

    return λs
end

function λmap(E; params=PhysicalParameters(), ic_alg=PoincareRand(n=500),
        ic_recompute=false, alg=DynSys(), recompute=false, root=(@__DIR__)*"/../../output/classical")
    g, t = @timed initialize(root)
    @debug "Loaded graph $g in $t seconds."
    λs = λmap!(g, E, params=params, ic_alg=ic_alg, ic_recompute=ic_recompute, alg=alg, recompute=recompute)
    remote_do(λhist, rand(workers()), λs, params, alg, E, ic_alg)
    _, t = @timed savechanges(g, root)
    @debug "Saving graph took $t seconds."

    return λs
end

function λ_timeseries_problem(p0::SVector{N}, q0::SVector{N}, alg::TimeRescaling;
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

function λ_timeseries_problem(z0::SVector{N}, alg::TimeRescaling;
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

function λproblem(z0::SVector{N}, alg::TimeRescaling;
        params=PhysicalParameters()) where {N}
    @unpack T, Ttr, d0, τ, solver, diff_eq_kwargs = alg

    idx1 = SVector{N}(1:N)
    idx2 = SVector{N}(N+1:2N)

    @inbounds dist(u) = norm(u[idx1] - u[idx2])

    function affect!(integrator)
        a = dist(integrator.u) / d0
        Δu = integrator.u[idx1] - integrator.u[idx2]
        integrator.u = vcat(integrator.u[idx1], integrator.u[idx1] + Δu/a)
    end

    function update_λ(affect!, integrator)
        a = dist(integrator.u) / d0
        affect![].saved_value += log(a)
        return nothing
    end

    λcb = ScalarSavingPeriodicCallback(affect!, update_λ, zero(eltype(z0)), τ)

    if Ttr ≠ 0
        prob_tr = parallel_problem(ż, [z0, z0.+d0/√N], Ttr, params)
        sol_tr = solve(prob_tr, solver; diff_eq_kwargs..., save_start=false, save_everystep=false)

        remake(prob_tr, tspan=T, callback=λcb, u0=sol_tr[end])
    else
        parallel_problem(ż, [z0, z0.+d0/√N], T, params, λcb)
    end
end

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

    function update_λ(affect!, integrator)
        a = dist(integrator.u) / d0
        affect![].saved_value += log(a)
        return nothing
    end

    λcb = ScalarSavingPeriodicCallback(affect!, update_λ, zero(eltype(p0)), τ)

    if Ttr ≠ 0
        prob_tr = parallel_problem(ṗ, q̇, [p0, p0.+d0/√(2N)], [q0, q0.+d0/√(2N)], Ttr, params)
        sol_tr = solve(prob_tr, solver; diff_eq_kwargs..., save_start=false, save_everystep=false)

        remake(prob_tr, tspan=T, callback=λcb, u0=sol_tr[end])
    else
        parallel_problem(ṗ, q̇, [p0, p0.+d0/√(2N)], [q0, q0.+d0/√(2N)], T, params, λcb)
    end
end

function output_λ(sol, i)
    sol.prob.callback.affect!.saved_value / sol.t[end], false
end

function output_λ_timeseries(sol, i)
    a = sol.prob.callback.discrete_callbacks[1].affect!.saved_values
    τ = a.t[2] - a.t[1]
    DiffEqArray([sum(a.saveval[1:i]) / (τ*i) for i in eachindex(a.saveval)], a.t), false
end

function λ_timeseries_map(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, alg::TimeRescaling;
        params=PhysicalParameters(), parallel_type=:none) where {N, T}
    λprob = λproblem(p0[1], q0[1], alg)
    prob_func(prob, i, repeat) = λproblem(p0[i], q0[i], alg)
    monte_prob = MonteCarloProblem(λprob, prob_func=prob_func,
        output_func=output_λ_timeseries)
    sim = solve(monte_prob, alg.solver; alg.diff_eq_kwargs..., num_monte=length(p0),
        save_start=false, save_everystep=false, parallel_type=parallel_type)
end

function λ_timeseries_map(z0::SVector{N, T}, alg::TimeRescaling;
        params=PhysicalParameters(), parallel_type=:none) where {N, T}
    λprob = λproblem(z0[1], alg)
    prob_func(prob, i, repeat) = λproblem(z0[i], alg)
    monte_prob = MonteCarloProblem(λprob, prob_func=prob_func,
        output_func=output_λ_timeseries)
    sim = solve(monte_prob, alg.solver; alg.diff_eq_kwargs..., num_monte=length(z0),
        save_start=false, save_everystep=false, parallel_type=parallel_type)
end

function λ_timeseries_map(p0, q0, alg::TimeRescaling; params::PhysicalParameters)
    p0 = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
    q0 = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
    if issplit(alg.solver)
        λ_timeseries_map(p0, q0, alg, params=params, parallel_type=:pmap)
    else
        z0 = [vcat(p0[i], q0[i]) for i ∈ axes(q0, 1)]
        λ_timeseries_map(z0, alg, params=params, parallel_type=:pmap)
    end
end

function λmap(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, alg::TimeRescaling;
        params=PhysicalParameters(), parallel_type=:none) where {N, T}
    λprob = λproblem(p0[1], q0[1], alg)
    prob_func(prob, i, repeat) = λproblem(p0[i], q0[i], alg)
    monte_prob = MonteCarloProblem(λprob, prob_func=prob_func,
        output_func=output_λ)
    sim = solve(monte_prob, alg.solver; alg.diff_eq_kwargs..., num_monte=length(p0),
        save_end=false, save_everystep=false, parallel_type=parallel_type)
end

function λmap(z0::Array{SVector{N, T}}, alg::TimeRescaling;
        params=PhysicalParameters(), parallel_type=:none) where {N, T}
    λprob = λproblem(z0[1], alg)
    prob_func(prob, i, repeat) = λproblem(z0[i], alg)
    monte_prob = MonteCarloProblem(λprob, prob_func=prob_func,
        output_func=output_λ)
    sim = solve(monte_prob, alg.solver; alg.diff_eq_kwargs..., num_monte=length(z0),
        save_end=false, save_everystep=false, parallel_type=parallel_type)
end

function λmap(p0, q0, alg::TimeRescaling; params::PhysicalParameters)
    p0 = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
    q0 = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
    z0 = [vcat(p0[i], q0[i]) for i ∈ axes(q0, 1)]
    λmap(z0, alg, params=params, parallel_type=:pmap).u
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

end  # module Lyapunov
