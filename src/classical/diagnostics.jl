module Diagnostics

export monte_err

using ..Parameters
using ..Hamiltonian
using ..ParallelTrajectories
using ..DInfty: parallel_evolution
using ..Lyapunov

using OrdinaryDiffEq
using DiffEqMonteCarlo
using RecursiveArrayTools
using StaticArrays

energy_err(sol, offset=0; p) =
    [abs.(H([sol[offset+1,j], sol[offset+2,j]], [sol[offset+3,j], sol[offset+4,j]],p) -
          H([sol[offset+1,1], sol[offset+2,1]], [sol[offset+3,1], sol[offset+4,1]],p)) for j in axes(sol,2)]

energy_err_split(sol, offset=0; p) =
    [abs.(H([sol[offset+1,j], sol[offset+2,j]], [sol[offset+5,j], sol[offset+6,j]],p) -
          H([sol[offset+1,1], sol[offset+2,1]], [sol[offset+5,1], sol[offset+6,1]],p)) for j in axes(sol,2)]

function monte_err(u0::Array{SVector{N, T}}, d0, t, dt=0.01;
        params=PhysicalParameters(), parallel_type=:none,
        save_start=true, save_everystep=true, alg=Vern9(),
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    function output_func(sol, i)
        E1 = energy_err(sol, p=params)
        E2 = energy_err(sol, 4, p=params)

        return (DiffEqArray(E1, sol.t), DiffEqArray(E2, sol.t)), false
    end

    parallel_evolution(u0, d0, t, dt, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=save_start, save_everystep=save_everystep,
        alg=alg, kwargs=kwargs)
end

function monte_err(p0::Array{SVector{N, T}}, q0::Array{SVector{N, T}}, d0, t, dt=0.01;
    params=PhysicalParameters(), parallel_type=:none,
    save_start=true, save_everystep=true, alg=DPRKN12(),
    kwargs=(abstol=1e-14, reltol=0, maxiters=1e9)) where {N, T}

    function output_func(sol, i)

        E1 = energy_err_split(sol, p=params)
        E2 = energy_err_split(sol, 2, p=params)

        return (DiffEqArray(E1, sol.t), DiffEqArray(E2, sol.t)), false
    end

    parallel_evolution(p0, q0, d0, t, dt, params=params, parallel_type=parallel_type,
        output_func=output_func, save_start=save_start, save_everystep=save_everystep,
        alg=alg, kwargs=kwargs)
end

function λ_parameter_variation(p0::SVector, q0::SVector,
        algs::Vector{TimeRescaling{R}}; params=PhysicalParameters(),
        parallel_type=:none) where {R}
    @unpack T, Ttr, d0, τ, solver, diff_eq_kwargs = algs[1]

    prob = λproblem(p0, q0, algs[1])
    prob_func(prob, i, repeat) = λproblem(p0, q0, algs[i])
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func)
    sim = solve(monte_prob, solver; diff_eq_kwargs..., num_monte=length(algs),
        save_start=false, save_everystep=false, parallel_type=parallel_type)

    λs = [Lyapunov.λ(sim[i]) for i ∈ eachindex(sim)]

    return λs
end

function λ_τ_variation(p0::SVector, q0::SVector, τs;
        T=1e4, Ttr=1e3, d0=1e-9, solver=DPRKN12(),
        kwargs=(abstol=1e-14, reltol=0, maxiters=1e9),
        params=PhysicalParameters(), parallel_type=:none)

    algs = [TimeRescaling(T, Ttr, τ, d0, solver, kwargs) for τ in τs]
    λs = λ_parameter_variation(p0, q0, algs, params=params,
        parallel_type=parallel_type)

    return [λ.u[end] for λ ∈ λs]
end

end  # module Diagnostics
