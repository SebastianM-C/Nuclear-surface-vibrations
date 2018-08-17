!contains(==, names(Main), :Hamiltonian) && @everywhere include("hamiltonian.jl")
!contains(==, names(Main), :InitialConditions) && include("initial_conditions.jl")

using Hamiltonian, InitialConditions
@everywhere using DynamicalSystemsBase, ChaosTools
using StaticArrays
using DataFrames, CSV

function λmap(E; A=1, B=0.55, D=0.4, n=15, m=15, T=10000., Ttr=3000., d0=1e-9,
               upper_threshold=1e-6, lower_threshold=1e-12,
               inittest = ChaosTools.inittest_default(4),
               dt=20., diff_eq_kwargs=Dict(:abstol=>1e-14, :reltol=>1e-14,
               :maxiters=>1e9), recompute=false)
    prefix = "../../output/classical/B$B-D$D/E$E"
    q0, p0, N = generateInitialConditions(E, n, m, params=(A,B,D))
    df = CSV.read("$prefix/z0.csv", allowmissing=:none, use_mmap=!is_windows())
    # Workaround for https://github.com/JuliaData/CSV.jl/issues/170
    if !haskey(df, :λs) || recompute
        λs = _λmap(q0, p0, N; A=A, B=B, D=D, T=T, Ttr=Ttr,
           d0=d0, upper_threshold=upper_threshold, lower_threshold=lower_threshold,
           inittest=inittest, dt=dt, diff_eq_kwargs=diff_eq_kwargs)
        df[:λs] = λs
        CSV.write("$prefix/z0.csv", df)
    else
        df = CSV.read("$prefix/z0.csv", allowmissing=:none)
        λs = df[:λs]
    end
    return λs
end

function _λmap(q0, p0, N; A=1, B=0.55, D=0.4, T=10000., Ttr=1000., d0=1e-9,
               upper_threshold=1e-6, lower_threshold=1e-12,
               inittest = ChaosTools.inittest_default(4),
               dt=10., diff_eq_kwargs=Dict(:abstol=>d0, :reltol=>d0))
    z0 = [SVector{4}(hcat(p0[i, :], q0[i, :])) for i=1:N]
    ds = DynamicalSystemsBase.ContinuousDynamicalSystem(ż, z0[1], (A,B,D))

    pinteg = DynamicalSystemsBase.parallel_integrator(ds,
            [deepcopy(DynamicalSystemsBase.get_state(ds)),
            inittest(DynamicalSystemsBase.get_state(ds), d0)];
            diff_eq_kwargs=diff_eq_kwargs)

    λs = pmap(1:N) do i
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
    prob_func(prob, i, repeat) = ttr≠0?DynamicalODEProblem(f1, f2, p0[i], q0[i], tspan, p, callback=rescale):dist_prob(p, p0[i], q0[i], d0, tspan, rescale)

    function output_func(sol, i)
        λᵢ = [log(dist(s)/d0)/τ for s in sol.u]
        n = length(sol.t)
        λ_series = DiffEqArray([sum(λᵢ[1:i]) for i in eachindex(λᵢ)]./(1:n), sol.t)
        return λ_series[end], false
    end
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func, output_func=output_func)
    solve(monte_prob, DPRKN12(); kwargs..., saveat=τ, num_monte=n, parallel_type=parallel_type)
end
