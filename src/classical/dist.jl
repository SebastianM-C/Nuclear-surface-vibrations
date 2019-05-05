module DInfty

export monte_dist, d∞, Γ, DInftyAlgorithm

using ..Distributed
using ..Parameters
using ..Hamiltonian
using ..DataBaseInterface
using ..InitialConditions
using ..InitialConditions: depchain, initial_conditions!
using ..ParallelTrajectories
using ..Classical: AbstractAlgorithm

using LinearAlgebra: norm
using OrdinaryDiffEq
using DiffEqMonteCarlo
using DiffEqCallbacks
using Plots, LaTeXStrings
using RecursiveArrayTools
using StaticArrays
using StorageGraphs
using LightGraphs: outneighbors

@with_kw struct DInftyAlgorithm{R <: Real} <: AbstractAlgorithm @deftype R
    T = 500.
    d0 = 1e-9
    solver::OrdinaryDiffEqAlgorithm = DPRKN12()
    diff_eq_kwargs::NamedTuple = (abstol=1e-14, reltol=1e-14, maxiters=1e9)
end

function dhist(d, params, alg, E, ic_alg)
    T = alg.T
    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    plt = histogram(d, nbins=50, xlabel=L"d\infty", ylabel=L"N", label="T = $T")
    fn = string(typeof(alg)) * "_T$T" * "_hist"
    fn = replace(fn, "{Float64}" => "")
    fn = replace(fn, "NuclearSurfaceVibrations.Classical.DInfty." => "")
    dir = "$prefix/"*string(typeof(ic_alg))
    dir = replace(dir, "NuclearSurfaceVibrations.Classical.InitialConditions." => "")
    if !isdir(dir)
        mkpath(dir)
    end
    @debug "Saving plot"
    savefig(plt, dir*"/dinf_$fn.pdf")
end

function computed∞(node, alg, ic_alg, ic_dep, params, E, g)
    q0 = node[:q0]
    p0 = node[:p0]
    d, t = @timed d∞(p0, q0, alg; params=params)
    @debug "Computing values took $t seconds."
    _, t = @timed add_nodes!(g, foldr(=>,(ic_dep..., node, (d∞_alg=alg,), (d∞=d,))))
    @debug "Adding d∞ to graph took $t seconds."

    return d
end

function monte_dist(z0::Array{SVector{N, T}}, d0, t;
        params=PhysicalParameters(), parallel_type=:none, alg=Vern9(),
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = DiffEqArray([norm(s[idx1] - s[idx2]) for s in sol.u], sol.t)
        return d, false
    end

    parallel_evolution(ż, z0, d0, t, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=save_start, save_everystep=save_everystep,
        saveat=saveat, alg=alg, kwargs=kwargs)
end

function monte_dist(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, d0, t;
        params=PhysicalParameters(), parallel_type=:none, alg=DPRKN12(),
        saveat=typeof(t)[], kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    dprob = dist_prob(p0[1], q0[1], d0, t, params=params, alg=alg, saveat=saveat)
    prob_func(prob, i, repeat) = dist_prob(p0[i], q0[i], d0, t, params=params, alg=alg, saveat=saveat)

    monte_prob = MonteCarloProblem(dprob, prob_func=prob_func,
        output_func=output_d)
    sim = solve(monte_prob, alg; kwargs..., num_monte=length(p0),
        save_start=false, save_everystep=false, parallel_type=parallel_type)
end

function dist_prob(p0::SVector{N}, q0::SVector{N}, d0, T; params=PhysicalParameters(),
        alg=DPRKN12(), saveat=typeof(T)[], save_start=true, save_everystep=isempty(saveat)) where {N}

    idx1 = SVector{N}(1:N)
    idx2 = SVector{N}(N+1:2N)
    @assert length(p0) == length(q0)

    @inbounds dist(u) =
        norm(vcat(u[1,:][idx1] - u[1,:][idx2], u[2,:][idx1] - u[2,:][idx2]))

    d = SavedValues(typeof(T), eltype(q0[1]))
    save_func(u, t, integrator) = dist(u)
    cb = SavingCallback(save_func, d, saveat=saveat, save_everystep=save_everystep,
        save_start=save_start)
    parallel_problem(ṗ, q̇, [p0, p0.+d0/√(2N)], [q0, q0.+d0/√(2N)], T, params, cb)
end

function output_d(sol, i)
    d = sol.prob.callback.affect!.saved_values
    DiffEqArray(d.saveval, d.t), false
end

function d∞(z0::Array{SVector{N, T}}, alg::DInftyAlgorithm,
        params=PhysicalParameters(); parallel_type=:none) where {N, T}

    @unpack d0, solver, diff_eq_kwargs = alg
    t = alg.T

    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = norm(sol[end][idx1] - sol[end][idx2])
        return d, false
    end
    parallel_evolution(ż, z0, d0, t, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=false, save_everystep=false,
        alg=solver, kwargs=diff_eq_kwargs)
end

function d∞(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, alg::DInftyAlgorithm,
        params=PhysicalParameters(); parallel_type=:none) where {N, T}

    @unpack d0, solver, diff_eq_kwargs = alg
    t = alg.T

    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = norm(vcat(sol[end][1,:][idx1] - sol[end][1,:][idx2],
                      sol[end][2,:][idx1] - sol[end][2,:][idx2]))
        return d, false
    end
    parallel_evolution(ṗ, q̇, p0, q0, d0, t, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=false, save_everystep=false,
        alg=solver, kwargs=diff_eq_kwargs)
end

function d∞(p0, q0, alg::DInftyAlgorithm; params::PhysicalParameters)
    p0 = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
    q0 = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
    d∞(p0, q0, alg, params, parallel_type=:pmap).u
end

function d∞!(g::StorageGraph, E; params=PhysicalParameters(), ic_alg=PoincareRand(n=500),
        ic_recompute=false, alg=DInftyAlgorithm(), recompute=false)
    ic_node = initial_conditions!(g, E, alg=ic_alg, params=params, recompute=ic_recompute)
    ic_dep = depchain(params, E, ic_alg)
    # ic_node is the node with the compatible initial conditions
    # we need to check if d∞ was computed for this node
    outn = outneighbors(g, g[ic_node])

    if length(outn) > 0 && any(has_prop.(Ref(g), outn, :d∞_alg))
        # we have some d∞ computed, we now have to check if they are the
        # right ones
        @debug "Looking for available d∞_algs"
        idx = findfirst(v->g[v] == (d∞_alg=alg,), outn)
        if idx !== nothing
            @debug "Found compatible d∞"
            vals, t = @timed g[:d∞, ic_dep..., (d∞_alg=alg,)]
            @assert length(vals) == 1 "d∞ values not uniquely represented by deps!"
            d = vals[1]
            @debug "Loading saved λs took $t seconds."
        else
            d = computed∞(ic_node, alg, ic_alg, ic_dep, params, E, g)
        end
    else
        d = computed∞(ic_node, alg, ic_alg, ic_dep, params, E, g)
    end

    return d
end

function d∞(E; params=PhysicalParameters(), ic_alg=PoincareRand(n=500),
        ic_recompute=false, alg=DInftyAlgorithm(), recompute=false, root=(@__DIR__)*"/../../output/classical")
    g, t = @timed initialize(root)
    @debug "Loaded graph $g in $t seconds."
    d = d∞!(g, E, params=params, ic_alg=ic_alg, ic_recompute=ic_recompute, alg=alg, recompute=recompute)
    remote_do(dhist, rand(workers()), d, params, alg, E, ic_alg)
    _, t = @timed savechanges(g, root)
    @debug "Saving graph took $t seconds."

    return d
end

Γ(λ, d∞) = (exp(λ) - 1) / d∞

end  # module DInfty
