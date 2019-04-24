using NuclearSurfaceVibrations
using IntervalArithmetic
using StorageGraphs
using .Classical
using Plots, StatsPlots
using Statistics
using DataFrames

E = 50.
p = PhysicalParameters(B=0.5)
ic_alg = PoincareRand(n=500)
ic_dep = Classical.InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()

@time g[:λ, ic_dep..., (λ_alg=DynSys(),)]
@time g[foldr(=>, (ic_dep..., (λ_alg=DynSys(),) ) )]


@time df = mean_over_ic(g, :λ, (λ_alg=DynSys(T=1e5),), ic_alg, p, 0..100)
@profiler Reductions.hist_mean(g[:λ, dep..., (λ_alg=DynSys(),)])

mean_over_ic(g, DynSys(T=1e6), ic_alg, params=p, Einterval=0..1000)
@time mean_over_E(g, :λ, (λ_alg=DynSys(T=1e5),), ic_alg, p, 0..400, 0.1:0.1:0.4)

@time mean_over_E(g, DynSys(T=1e5), 0..400, ic_alg=ic_alg, Binterval=0.1:0.1:0.4)
