module DInfty

export d∞, Γ, DInftyAlgorithm

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

function parallel_evolution(u0, t;
        params=PhysicalParameters(), parallel_type=:none,
        alg=Vern9(), output_func=(sol, i)->(sol, false), cb=nothing,
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9))

    n = length(u0)
    tspan = (zero(typeof(t)), t)
    prob = parallel_problem(ż, u0[1], tspan, params, cb)

    prob_func(prob, i, repeat) = parallel_problem(ż, u0[i], tspan, params, cb)
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    sim = solve(monte_prob, alg; kwargs..., saveat=dt, num_monte=n,
        save_start=save_start, save_everystep=save_everystep,
        parallel_type=parallel_type)

    return sim
end

function parallel_evolution(z0::Array{SVector{N, T}}, d0, t;
        params=PhysicalParameters(), parallel_type=:none,
        alg=Vern9(), output_func=(sol, i)->(sol, false), cb=nothing,
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    u0 = [(z, z.+d0/√N) for z ∈ z0]
    parallel_evolution(u0, t, params=params, parallel_type=parallel_type,
        alg=alg, output_func=output_func, cb=cb, save_start=save_start,
        save_everystep=save_everystep, saveat=saveat, kwargs=kwargs)
end

function parallel_evolution(u1, u2, t;
        params=PhysicalParameters(), parallel_type=:none,
        alg=DPRKN12(), output_func=(sol, i)->(sol, false), cb=nothing,
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9))

    @assert length(u1) == length(u2)
    n = length(u1)
    tspan = (zero(typeof(t)), t)
    prob = parallel_problem(ṗ, q̇, u1[1], u2[1], tspan, params, cb)

    prob_func(prob, i, repeat) = parallel_problem(ṗ, q̇, u1[i], u2[i], tspan, params, cb)
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    sim = solve(monte_prob, alg; kwargs..., saveat=dt, num_monte=n,
        save_start=save_start, save_everystep=save_everystep,
        parallel_type=parallel_type)

    return sim
end

function parallel_evolution(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}},
        d0, t; params=PhysicalParameters(), parallel_type=:none,
        alg=DPRKN12(), output_func=(sol, i)->(sol, false), cb=nothing,
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    u1 = [(p[i], p[i].+d0/√(2N)) for p ∈ p0]
    u2 = [(q[i], q[i].+d0/√(2N)) for q ∈ q0]

    parallel_evolution(u1, u2, t, params=params, parallel_type=parallel_type,
        alg=alg, output_func=output_func, cb=cb, save_start=save_start,
        save_everystep=save_everystep, saveat=saveat, kwargs=kwargs)
end

function monte_dist(u0::Array{SVector{N, T}}, d0, t;
        params=PhysicalParameters(), parallel_type=:none, alg=Vern9(),
        save_start=true, save_everystep=true, saveat=typeof(t)[],
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = DiffEqArray([norm(s[idx1] - s[idx2]) for s in sol.u], sol.t)
        return d, false
    end

    parallel_evolution(u0, d0, t, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=save_start, save_everystep=save_everystep,
        saveat=saveat, alg=alg, kwargs=kwargs)
end

function monte_dist(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, d0, t;
    params=PhysicalParameters(), parallel_type=:none, alg=DPRKN12(),
    save_start=true, save_everystep=true, saveat=typeof(t)[],
    kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = DiffEqArray([norm(vcat(s[1,:][idx1] - s[1,:][idx2],
                                   s[2,:][idx1] - s[2,:][idx2])) for s in sol.u],sol.t)
        return d, false
    end

    parallel_evolution(p0, q0, d0, t, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=save_start, save_everystep=save_everystep,
        saveat=saveat, alg=alg, kwargs=kwargs)
end

function d∞(z0::Array{SVector{N, T}}, alg::DInftyAlgorithm;
    params=PhysicalParameters(), parallel_type=:none) where {N, T}

    @unpack d0, solver, diff_eq_kwargs = alg
    t = alg.T

    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = norm(sol[end][idx1] - sol[end][idx2])
        return d, false
    end
    parallel_evolution(z0, d0, t, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=false, save_everystep=false,
        alg=solver, kwargs=diff_eq_kwargs)
end

function d∞(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, alg::DInftyAlgorithm;
    params=PhysicalParameters(), parallel_type=:none) where {N, T}

    @unpack d0, solver, diff_eq_kwargs = alg
    t = alg.T

    function output_func(sol, i)
        idx1 = SVector{N}(1:N)
        idx2 = SVector{N}(N+1:2N)

        d = norm(vcat(sol[end][1,:][idx1] - sol[end][1,:][idx2],
                      sol[end][2,:][idx1] - sol[end][2,:][idx2]))
        return d, false
    end
    parallel_evolution(p0, q0, d0, t, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=false, save_everystep=false,
        alg=solver, kwargs=diff_eq_kwargs)
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

    if !recompute && count(skipmissing(cond)) == count(ic_cond)
        _cond = BitArray(replace(cond, missing=>false))
        @debug "Loading compatible values."
        d = unique(db.df[_cond, :d∞])
    else
        @debug "Incompatible values. Computing new values."
        q0 = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
        p0 = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
        d = d∞(q0, p0, alg; params=params, parallel_type=parallel_type).u
        df = build_df(d, alg)

        if !haskey(db.df, :d∞)
            db.df[:d∞] = Array{Union{Missing, Float64}}(fill(missing, size(db.df, 1)))
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
