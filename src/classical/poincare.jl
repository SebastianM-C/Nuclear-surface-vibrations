#!/usr/bin/env julia
using DifferentialEquations
using Plots
using ArgParse
using ParallelDataTransfer

include("$(pwd())/initial_conditions.jl")
using Hamiltonian, InitialConditions

pyplot()

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
    "--energy", "-e"
         help = "The energy of the system"
         arg_type = Float64
         # nargs = '+'
         default = 120.
    "-n"
        help = "Target number of initial conditions per kinetic energy value"
        arg_type = Int64
        default = 15
    "-m"
        help = "Target number of kinetic energy showvalues"
        arg_type = Int64
        default = 15
    "-t"
        help = "Simulation duration"
        arg_type = Float64
        default = 800.
    end
    parsed_args = parse_args(ARGS, arg_settings)
    E = parsed_args["energy"]
    n = parsed_args["n"]
    m = parsed_args["m"]
    t = parsed_args["t"]

    E, n, m, t
end

@everywhere begin
condition(t, u, integrator) = u
affect!(integrator) = nothing
PoincareCb(idx) = DifferentialEquations.ContinuousCallback(condition, affect!,
    nothing, save_positions=(false, true), idxs=idx)
end

# function g(u, resid)
#     resid[1,1] = H(u[1,:], u[2,:]) - E
#     resid[1,2] = 0
#     resid[2,:] .= 0
# end

function energy_error(sim, E)
    energy_err(t,u1,u2,u3,u4) = (t, H([u1,u2],[u3,u4]) - E)
    energy_err(sol) = size(sol.u,1) > 0 ?
        [abs.(H([sol[1,j], sol[2,j]], [sol[3,j], sol[4,j]]) - E)
        for j=1:size(sol,2)] : 0
    info("The maximum energy error during time evolution was "*
        "$(maximum(map(i->maximum(energy_err(i)), sim.u)))")

    plt = plot(ylabel="Energy error", legend=false)
    scatter!(plt, sim, vars=(energy_err, 0,1,2,3,4), msc=nothing, ms=2)
end



# function main()

E, n, m, t = input_param()
tspan = (0., t)
q0, p0, N = generateInitialConditions(E, n, m)

prob = HamiltonianProblem(H, q0[1,:], p0[1,:], tspan)
# prob = DynamicalODEProblem(q̇, ṗ, q0[1,:], p0[1,:], tspan)
@everywhere cb = PoincareCb(1)
sendto(workers(), prob=prob, q0=q0, p0=p0)

@everywhere function prob_func(prob, i, repeat)
    # mp = ManifoldProjection(g, nlopts=Dict(:ftol=>1e-12))
    # cbs = CallbackSet(cb, mp)
    DifferentialEquations.ODEProblem(prob.f,
        (q0[i,:], p0[i,:]), prob.tspan, callback=cb)
end

monte_prob = MonteCarloProblem(prob, prob_func=prob_func)

sim = solve(monte_prob, Vern9(), abstol=1e-14, reltol=1e-14,
    save_everystep=false, save_start=false, save_end=false, num_monte=N,
    parallel_type=:pmap)
# N = 21:   120s  serial vs 128s threaded (20)
# N = 211:  1271s serial vs 549s pmap     (2)
# N = 211:  1271s serial vs 124s,118s,102s pmap     (20)

plt = energy_error(sim, E)

plt2 = scatter(sim, vars=(2,4), msc=nothing, ms=2.)

# end

# main()

# using RecipesBase
# @recipe function f(sim::AbstractMonteCarloSolution;
#                    idxs = typeof(sim.u)<:AbstractArray ? eachindex(sim.u) : 1)
#   for i in idxs
#     size(sim[i].u, 1) == 0 && continue
#     @series begin
#       legend := false
#       xlims --> (-Inf,Inf)
#       ylims --> (-Inf,Inf)
#       zlims --> (-Inf,Inf)
#       sim[i]
#     end
#   end
# end
