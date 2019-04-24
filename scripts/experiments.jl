using Distributed

@progress for i in 2:8
    addprocs([("cn$i", 40)], exename="/mnt/storage/julia.sh")
end

addprocs(12)
addprocs([("headnode:35011", 40)], exename="/mnt/storage/julia.sh",
    tunnel=true, dir="/mnt/storage/Nuclear-surface-vibrations")

@time using NuclearSurfaceVibrations
using .Classical
using StorageGraphs
using StaticArrays
using OrdinaryDiffEq
using DiffEqMonteCarlo
using DiffEqParamEstim
using Plots
using Statistics
using Optim

E = 10.
ic_alg = PoincareRand(n=500)
p = PhysicalParameters()
alg = TimeRescaling(τ=5.)
@time g = initialize()

q0, p0 = initial_conditions(g, E, alg=ic_alg)
@time λmap(p0, q0, DynSys(), params=p)
@time λmap(p0, q0, TimeRescaling(), params=p)

p₀ = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
q₀ = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
z0 = [vcat(p₀[i], q₀[i]) for i ∈ axes(q₀, 1)]
λprob = λproblem(z0[1], alg)
prob_func(prob, i, repeat) = λproblem(z0[i], alg)
monte_prob = MonteCarloProblem(λprob, prob_func=prob_func, output_func=Classical.Lyapunov.output_λ)
sim = solve(monte_prob, Vern9(), num_monte=30, parallel_type=:none)
histogram(sim.u)
function loss(sim; nbins=50)
    tot_loss = 0.
    if any((!(s isa Number) for s in sim))
        tot_loss = Inf
    else
        data = Reductions.select_after_first_max(sim.u, nbins=nbins)
        tot_loss = std(data)
    end
    tot_loss
end

loss(sim)
std(sim.u)
function prob_generator(prob, p)
    λprob = λproblem(z0[1], TimeRescaling(τ=p[1]))
    prob_func(prob, i, repeat) = λproblem(z0[i], alg)
    MonteCarloProblem(λprob, prob_func=prob_func, output_func=Classical.Lyapunov.output_λ)
end

obj = build_loss_objective(monte_prob, Vern9(), loss,
    prob_generator=prob_generator, num_monte=30, parallel_type=:none)

optimize(obj, 0.5, 50.)
