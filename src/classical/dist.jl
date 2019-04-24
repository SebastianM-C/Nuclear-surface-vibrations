module DInfty

export monte_dist, d∞, Γ, DInftyAlgorithm

using ..Distributed
using ..Parameters
using ..Hamiltonian
using ..DataBaseInterface
using ..InitialConditions
using ..InitialConditions: depchain, initial_conditions!, extract_ics
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

function computed∞(nodes, alg, ic_alg, ic_deps, params, E, g)
    q0, p0 = extract_ics(nodes, ic_alg)
    d, t = @timed d∞(p0, q0, alg; params=params)
    @debug "Computing values took $t seconds."
    _, t = @timed add_derived_values!(g, ic_deps, (q₀=q0[:,1],q₂=q0[:,2], p₀=p0[:,1],p₂=p0[:,2]), (d∞=d,), (d∞_alg=alg,))
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
    ic_nodes = initial_conditions!(g, E, alg=ic_alg, params=params, recompute=ic_recompute)
    ic_deps = depchain(params, E, ic_alg)
    # ic_vertices are all the compatible initial conditions
    # we need to check that for each initial condition all the
    # outneighbors have at least a λ computed
    ic_vertices, t = @timed get.(Ref(g.index), ic_nodes, 0)
    @debug "Found $(length(ic_vertices)) compatible initial conditions in $t seconds"
    if mapreduce(v->any(has_prop.(Ref(g), outneighbors(g, v), :d∞_alg)), &, ic_vertices)
        # we have some d∞ computed, we now have to check if they are the
        # right ones
        @debug "Looking for available d∞_algs"
        d∞_algs = Int[]
        for v in ic_vertices
            union!(d∞_algs, outneighbors(g, v))
        end
        @debug "Checking for compatible d∞_algs"
        if mapreduce(a->g[a] == (d∞_alg=alg,), |, d∞_algs)
            @debug "Found compatible d∞"
            d, t = @timed g[:d∞, ic_deps..., (d∞_alg=alg,)]
            @debug "Loading saved d∞ took $t seconds."
        else
            d = computed∞(ic_nodes, alg, ic_alg, ic_deps, params, E, g)
        end
    else
        d = computed∞(ic_nodes, alg, ic_alg, ic_deps, params, E, g)
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
