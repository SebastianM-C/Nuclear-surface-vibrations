using Distributed

addprocs(12)
addprocs([("headnode:35011", 40)], exename="/mnt/storage/julia.sh",
    tunnel=true, dir="/mnt/storage/Nuclear-surface-vibrations")

@time using NuclearSurfaceVibrations
using .Classical
using .InitialConditions: depchain, initial_conditions!
using StorageGraphs
using .Visualizations
using OrdinaryDiffEq
using StaticArrays
using AbstractPlotting
using StatsBase
using LinearAlgebra:norm

E = 0.1
p = PhysicalParameters(B=0.15)
ic_alg = PoincareRand(n=10)
ic_dep = depchain(p,E,ic_alg)
@time g = initialize()

@time poincare_explorer(g, E, DynSys(), ic_alg, params=p)
q0, p0 = initial_conditions!(g, E, alg=ic_alg, params=p)
sim = poincaremap(q0, p0, params=p, t=100)
l = Lyapunov.λmap!(g, E, params=p, ic_alg=ic_alg, alg=DynSys())
# fit(Histogram, l, closed=:left, nbins=20)

q0, p0 = initial_conditions!(g, E, alg=ic_alg, params=p)
p₀ = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
q₀ = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
z0 = [vcat(p₀[i], q₀[i]) for i ∈ axes(q₀, 1)]
prob = ODEProblem(ż, z0[1], 100., p)
sol = solve(prob, Vern9(), abstol=1e-14, reltol=1e-14, maxiters=1e9)

t = Node(0.)
surface_sc = animate_solution(sol, t)
section_sc = θϕ_sections(sol, t, surface_sc.limits[])
line_sc = path_animation3D(sol, t)
plot_slice!(line_sc, sim[1])

vbox(surface_sc, line_sc, sizes=[0.5,0.5])
vbox(surface_sc, section_sc, sizes=[0.7, 0.3])

Visualizations.animate(t, (0, 40))


Point3f0(sol(0, idxs=[1,2,3]))
using .Classical.ParallelTrajectories

d0 = 1e-3
pprob = parallel_problem(ż, (z0[1], z0[1].+d0/√4), 40., p)
psol = solve(pprob, Vern9(), abstol=1e-14, reltol=1e-14, maxiters=1e9)

parallel_paths(psol, t)

λprob = λ_timeseries_problem(z0[1], TimeRescaling(T=100.,Ttr=0.), params=p)
λsol = solve(λprob, Vern9(), abstol=1e-14, reltol=1e-14, maxiters=1e9)

parallel_sc = parallel_paths(psol, t)
psc = paths_distance(psol, t)
lpsc = paths_distance_log(psol, t)
dsc = hbox(psc, lpsc)
sc = vbox(parallel_sc, dsc, sizes=[0.5, 0.5])

vbox(paths_distance(λsol, t),paths_distance_log(λsol, t))
save_animation(sc, t, (0, 40), "output/classical/test.webm")
save("output/classical/nucleus.png", surface_sc)
