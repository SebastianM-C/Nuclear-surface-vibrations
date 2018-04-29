#!/usr/bin/env julia
addprocs(4)
using DifferentialEquations
using Plots
# using ArgParse
using ParallelDataTransfer

include("initial_conditions.jl")
using Hamiltonian, InitialConditions

# pyplot()
#
# function input_param()
#     arg_settings = ArgParseSettings()
#     @add_arg_table arg_settings begin
#     "--energy", "-e"
#          help = "The energy of the system"
#          arg_type = Float64
#          # nargs = '+'
#          default = 120.
#     "-n"
#         help = "Target number of initial conditions per kinetic energy value"
#         arg_type = Int64
#         default = 15
#     "-m"
#         help = "Target number of kinetic energy showvalues"
#         arg_type = Int64
#         default = 15
#     "-t"
#         help = "Simulation duration"
#         arg_type = Float64
#         default = 800.
#     end
#     parsed_args = parse_args(ARGS, arg_settings)
#     E = parsed_args["energy"]
#     n = parsed_args["n"]
#     m = parsed_args["m"]
#     t = parsed_args["t"]
#
#     E, n, m, t
# end

# @everywhere begin
condition(u, t, integrator) = u
affect!(integrator) = nothing
PoincareCb(idx) = ContinuousCallback(condition, affect!, nothing,
    save_positions=(false, true), idxs=idx)
# end
# function g(u, resid)
#     resid[1,1] = H(u[1,:], u[2,:]) - E
#     resid[1,2] = 0
#     resid[2,:] .= 0
# end

function energy_error(sim, E, params)
    energy_err(t,u1,u2,u3,u4) = (t, H([u1,u2],[u3,u4], params) - E)
    energy_err(sol) = size(sol.u,1) > 0 ?
        [abs.(H([sol[1,j], sol[2,j]], [sol[3,j], sol[4,j]], params) - E)
        for j=1:size(sol,2)] : 0
    info("The maximum energy error during time evolution was "*
        "$(maximum(map(i->maximum(energy_err(i)), sim.u)))")

    plt = plot(ylabel="Energy error", legend=false)
    plot!(plt, sim, vars=(energy_err, 0,1,2,3,4), msc=nothing, ms=2)
end


A, B, D = 1, 0.55, 0.4
E, n, m = 30., 10, 10
tspan = (0., 500.)
q0, p0, N = generateInitialConditions(E, n, m, A, B, D)
z0 = hcat(p0, q0)
# prob = HamiltonianProblem(H, q0[1,:], p0[1,:], tspan)
# prob = DynamicalODEProblem(q̇, ṗ, q0[1,:], p0[1,:], tspan)
prob = ODEProblem(ż, z0[1, :], tspan, (A, B, D), callback=cb)
@everywhere cb = PoincareCb(3)
sendto(workers(), prob=prob, z0=z0)

@everywhere function prob_func(prob, i, repeat)
    # mp = ManifoldProjection(g, nlopts=Dict(:ftol=>1e-12))
    # cbs = CallbackSet(cb, mp)
    DifferentialEquations.ODEProblem(prob.f,
        z0[i, :], prob.tspan, prob.p, callback=cb)
end

monte_prob = MonteCarloProblem(prob, prob_func=prob_func)

sim = solve(monte_prob, Vern9(), abstol=1e-14, reltol=1e-14,
    save_everystep=false, save_start=false, save_end=false,
    save_everystep=false, num_monte=30, parallel_type=:threads)
# N = 21:   120s  serial vs 128s threaded (20)
# N = 211:  1271s serial vs 549s pmap     (2)
# N = 211:  1271s serial vs 124s,118s,102s pmap     (20)

plt = energy_error(sim, E, (A, B, D))

plt2 = scatter(sim, vars=(4, 2), msc=nothing, ms=2.)
