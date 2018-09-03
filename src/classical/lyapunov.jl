module Lyapunov

export λmap, LyapunovAlgorithm, DynSys

using ..Distributed
using ..Parameters
using ..InitialConditions
using ..Classical: AbstractAlgorithm

using ChaosTools

using OrdinaryDiffEq
using StaticArrays

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

function build_df(λs, alg)
    N = size(λs, 1)
    # T, Ttr, d0, upper_threshold, lower_threshold, dt, solver, diff_eq_kwargs = unpack_with_missing(alg)
    @unpack T, Ttr, d0, upper_threshold, lower_threshold, dt, solver, diff_eq_kwargs = alg
    df = DataFrame()
    df[:λs] = categorical(λs)
    df[:lyap_alg] = categorical(fill(string(typeof(alg)), N))
    df[:lyap_T] = categorical(fill(T, N))
    df[:lyap_Ttr] = categorical(fill(Ttr, N))
    df[:lyap_d0] = categorical(fill(d0, N))
    df[:lyap_ut] = categorical(fill(upper_threshold, N))
    df[:lyap_lt] = categorical(fill(lower_threshold, N))
    df[:lyap_dt] = categorical(fill(dt, N))
    df[:lyap_integ] = categorical(fill("$solver", N))
    df[:lyap_diffeq_kw] = categorical(fill("$diff_eq_kwargs", N))
    allowmissing!(df)

    return df
end

function λmap(E; params=(A=1, B=0.55, D=0.4), alg=PoincareRand(n=500),
        recompute=false, lyapunov_alg=DynSys(), alg_recompute=false)

    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    q0, p0 = initial_conditions(E, n, m, params=(A,B,D), alg=alg,
        symmetric=symmetric, border_n=border_n, recompute=recompute)
    db = DataBase(E, params)
    n, m, border_n = unpack_with_missing(alg)
    ic_vals = Dict([:n, :m, :E, :initial_cond_alg, :border_n] .=>
                [n, m, E, string(typeof(alg)), border_n])
    ic_df, ic_cond = compatible(db.df, ic_vals)

    vals = Dict([:lyap_alg, :lyap_T, :lyap_Ttr, :lyap_d0, :lyap_ut,
                :lyap_lt, :lyap_dt, :lyap_integ, :lyap_diffeq_kw] .=>
                [string(typeof(lyapunov_alg)), T, Ttr, d0, upper_threshold,
                lower_threshold, dt, "$solver", "$diff_eq_kwargs"])
    filtered_df, λcond = compatible(db.df, vals)
    cond = ic_cond .& λcond
    compat = count(cond) > 0 && !recompute

    if compat
        λs = unique(filtered_df[:λs])
    else
        @debug "Incompatible values. Computing new values."
        λs = _λmap(q0, p0, alg; params=params)
        df = build_df(λs, alg)

        update!(db, df, cond)

        # plots
    end
    return λs
end

function _λmap(q0, p0, alg::DynSys; params=PhysicalParameters())
    @unpack T, Ttr, d0, upper_threshold, lower_threshold, dt, solver, diff_eq_kwargs = alg
    inittest = ChaosTools.inittest_default(4)
    z0 = [SVector{4}(vcat(p0[i, :], q0[i, :])) for i ∈ axes(q0, 1)]
    ds = ContinuousDynamicalSystem(ż, z0[1], params)

    pinteg = DynamicalSystemsBase.parallel_integrator(ds,
            [deepcopy(DynamicalSystemsBase.get_state(ds)),
            inittest(DynamicalSystemsBase.get_state(ds), d0)]; alg=solver,
            diff_eq_kwargs...)

    λs = pmap(eachindex(z0)) do i
            set_state!(pinteg, z0[i])
            reinit!(pinteg, pinteg.u)
            ChaosTools._lyapunov(pinteg, T, Ttr, dt, d0,
                upper_threshold, lower_threshold)
        end

    return λs
end

function λ_time(p, p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, d0=1e-9, t=1e4, ttr=100., τ=5.;
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
