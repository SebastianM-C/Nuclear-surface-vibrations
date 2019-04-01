module DInfty

export monte_dist, d∞, Γ, DInftyAlgorithm

using ..Distributed
using ..Parameters
using ..Hamiltonian
using ..DataBaseInterface
using ..InitialConditions
using ..ParallelTrajectories
using ..Classical: AbstractAlgorithm

using LinearAlgebra: norm
using OrdinaryDiffEq
using DiffEqMonteCarlo
using DiffEqCallbacks
using Plots, LaTeXStrings
using RecursiveArrayTools
using StaticArrays
using DataFrames

@with_kw struct DInftyAlgorithm{R <: Real} <: AbstractAlgorithm @deftype R
    T = 500.
    d0 = 1e-9
    solver::OrdinaryDiffEqAlgorithm = DPRKN12()
    diff_eq_kwargs::NamedTuple = (abstol=1e-14, reltol=1e-14, maxiters=1e9)
end

function build_df(d∞, alg)
    N = length(d∞)
    @unpack T, d0, solver, diff_eq_kwargs = alg
    df = DataFrame()
    df[:d∞] = categorical(d∞)
    df[:dinf_alg] = categorical(fill(string(typeof(alg)), N))
    df[:dinf_T] = categorical(fill(T, N))
    df[:dinf_d0] = categorical(fill(d0, N))
    df[:dinf_integ] = categorical(fill("$solver", N))
    df[:dinf_diffeq_kw] = categorical(fill("$diff_eq_kwargs", N))
    allowmissing!(df)

    return df
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

function d∞(E; params=PhysicalParameters(), ic_alg=PoincareRand(n=500),
        ic_recompute=false, alg=DInftyAlgorithm(), recompute=false,
        parallel_type=:pmap)
    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    q0, p0 = initial_conditions(E, alg=ic_alg, params=params, recompute=ic_recompute)
    db = DataBase(params)
    n, m, border_n = unpack_with_nothing(ic_alg)
    ic_vals = Dict([:n, :m, :E, :initial_cond_alg, :border_n] .=>
                [n, m, E, string(typeof(ic_alg)), border_n])
    ic_cond = compatible(db.df, ic_vals)
    @debug "Initial conditions compat" ic_cond
    @unpack T, d0, solver, diff_eq_kwargs = alg

    vals = Dict([:dinf_alg, :dinf_T, :dinf_d0, :dinf_integ, :dinf_diffeq_kw] .=>
                [string(typeof(alg)), T, d0, "$solver", "$diff_eq_kwargs"])
    dcond = compatible(db.df, vals)
    cond = ic_cond .& dcond
    @debug "Stored values compat" dcond cond

    if !recompute && count(skipmissing(cond)) == size(q0, 1)
        _cond = BitArray(replace(cond, missing=>false))
        @debug "Loading compatible values."
        d = unique(db.df[_cond, :d∞])
    else
        @debug "Incompatible values. Computing new values."
        d = d∞(q0, p0, alg; params=params)
        df = build_df(d, alg)

        if !haskey(db.df, :d∞)
            db.df[:d∞] = Array{Union{Missing, Float64}}(fill(missing, size(db.df, 1)))
        end

        if size(q0, 1) < count(ic_cond)
            @debug "Removing clones." size(q0, 1) count(ic_cond)
            # then we have to continue with only on set of initial conditions
            # clones can only appear because of a quantity that was computed
            # with different parameters and all the clones have the same size
            if recompute
                ic_cond .&= dcond
            end
            # we will keep only the first clone
            ic_cond = replace(ic_cond[end:-1:1], true=>false,
                count=count(ic_cond)-size(q0, 1))[end:-1:1]

        end

        update!(db, df, ic_cond, vals)

        plt = histogram(d, nbins=50, xlabel=L"d_\infty", ylabel=L"N", label="T = $T")
        fn = string(typeof(alg)) * "_T$T" * "_hist"
        fn = replace(fn, "{Float64}" => "")
        savefig(plt, "$prefix/"*string(typeof(ic_alg))*"/dinf_$fn.pdf")
    end
    arr_type = nonnothingtype(eltype(d))
    return disallowmissing(Array{arr_type}(d))
end

Γ(λ, d∞) = (exp(λ) - 1) / d∞

end  # module DInfty
