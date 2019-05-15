using NuclearSurfaceVibrations
using IntervalArithmetic
using StorageGraphs
using .Classical
using Plots, StatsPlots
using Statistics, StatsBase
using DataFrames
using Query

E = 0.1
p = PhysicalParameters(B=0.22)
ic_alg = PoincareRand(n=500)
ic_dep = InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()

@time g[:λ, ic_dep..., (λ_alg=DynSys(T=1e5),)][1]
@time g[foldr(=>, (ic_dep..., (λ_alg=DynSys(),) ) )]

@profiler Reductions.hist_mean(g[:λ, dep..., (λ_alg=DynSys(),)])

df = mean_over_ic(g, :λ, (λ_alg=DynSys(T=1e5),), ic_alg, p, 0.01:0.01:10)

mean_over_ic(g, DynSys(T=1e5), ic_alg, params=p, Einterval=0.01:0.01:10)
# mean_over_ic(g, TimeRescaling(T=1e5), ic_alg, params=p, Einterval=10:10:1000)
mean_over_ic(g, DInftyAlgorithm(T=1e5), ic_alg, params=p, Einterval=10:10:1000)
@time mean_over_E(g, :λ, (λ_alg=DynSys(T=1e5),), ic_alg, p, 0..400, 0.1:0.1:0.4)

@time mean_over_E(g, DynSys(T=1e5), 0.01:0.01:10, ic_alg=ic_alg)
