using NuclearSurfaceVibrations
using IntervalArithmetic
using StorageGraphs
using .Classical
using Plots, StatsPlots
using Statistics, StatsBase
using DataFrames
using Query
using LaTeXStrings


E = 3
p = PhysicalParameters(B=0.5)
ic_alg = PoincareRand(n=500)
ic_dep = InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()

@time g[:λ, ic_dep..., (λ_alg=DynSys(T=1e5),)][1]
@time g[foldr(=>, (ic_dep..., (λ_alg=DynSys(),) ) )]

l = g[:λ, ic_dep..., (λ_alg=DynSys(T=1e5),)][1]
d = g[:d∞, ic_dep..., (d∞_alg=DInftyAlgorithm(T=1e5),)][1]
l1, l2 = Visualizations.selected_hist(g, E, DynSys(T=1e5), ic_alg, params=p)
plot(l1);plot!(l2)

d1, d2 = Visualizations.selected_hist(g, E, DInftyAlgorithm(T=1e5), ic_alg,
    params=p, select=x->Reductions.select_after_first_max(x,ut=1))
plot(d1);plot!(d2)

histogram(l, nbins=50, xlabel=L"\lambda", ylabel=L"N", label="E=$E")
histogram(d, nbins=50, xlabel=L"d_\infty", ylabel=L"N", label="E=$E")

g = Γ.(l,d)
histogram(g, nbins=50)

@profiler g[foldr(=>, (ic_dep..., (λ_alg=DynSys(T=1e5),) ) )]
@time df = mean_over_ic(g, :λ, (λ_alg=DynSys(T=1e5),), ic_alg, p, 0..10)
@profiler Reductions.hist_mean(g[:λ, dep..., (λ_alg=DynSys(),)])
df |> @map({_.E, λ=_.val}) |> @orderby(_.E) |> @take(10) |> @df plot(:E, :λ)
DataFrame |> @df plot(:E, :λ)

df = mean_over_ic(g, :λ, (λ_alg=DynSys(T=1e5),), ic_alg, p, 0.01:0.01:10)

mean_over_ic(g, DynSys(T=1e5), ic_alg, params=p, Einterval=0.01:0.01:10)
# mean_over_ic(g, TimeRescaling(T=1e5), ic_alg, params=p, Einterval=10:10:1000)
mean_over_ic(g, DInftyAlgorithm(T=1e5), ic_alg, params=p, Einterval=10:10:1000)
@time mean_over_E(g, :λ, (λ_alg=DynSys(T=1e5),), ic_alg, p, 0..400, 0.1:0.1:0.4)

@time mean_over_E(g, DynSys(T=1e5), 0.01:0.01:10, ic_alg=ic_alg)
