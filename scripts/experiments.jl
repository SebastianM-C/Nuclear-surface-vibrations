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

E = 10.
ic_alg = PoincareRand(n=500)
p = PhysicalParameters()
alg = TimeRescaling()
@time g = initialize()

q0, p0 = initial_conditions(g, E, alg=ic_alg)
@time λmap(p0, q0, DynSys(), params=p)
@time λmap(p0, q0, TimeRescaling(), params=p)

p₀ = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
q₀ = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
λprob = λproblem(p₀[1], q₀[1], alg)
@everywhere prob_func(prob, i, repeat) = λproblem(p₀[i], q₀[i], alg)
monte_prob = MonteCarloProblem(λprob, prob_func=prob_func, output_func=Classical.Lyapunov.output_λ)
# sim = solve(monte_prob, Vern9(), num_monte=500, parallel_type=:threads)


obj = build_loss_objective(monte_prob, Vern9(), L2Loss(collect(0:200), ))
