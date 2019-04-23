using NuclearSurfaceVibrations
using IntervalArithmetic
using StorageGraphs
using .Classical
using Plots, StatsPlots
using Statistics

E = 50.
p = PhysicalParameters(B=0.55)
ic_alg = PoincareRand(n=500)
dep = Classical.InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()

@time g[:λ, dep..., (λ_alg=DynSys(),)]
@time g[foldr(=>, (dep..., (λ_alg=DynSys(),) ) )]


@time df = mean_over_ic(g, :λ, (λ_alg=DynSys(),), ic_alg, p, 0..500)
@profiler Reductions.hist_mean(g[:λ, dep..., (λ_alg=DynSys(),)])

mean_over_ic(g, DynSys(), ic_alg, params=p, Einterval=0..1000)
