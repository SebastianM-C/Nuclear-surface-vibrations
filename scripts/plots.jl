using Distributed

addprocs(12)
addprocs([("headnode:35011", 40)], exename="/mnt/storage/julia.sh",
    tunnel=true, dir="/mnt/storage/Nuclear-surface-vibrations")

@time using NuclearSurfaceVibrations
using .Classical
using StorageGraphs
using .Visualizations
using OrdinaryDiffEq
using StaticArrays
using AbstractPlotting

E = 0.4
p = PhysicalParameters(B=0.55)
ic_alg = PoincareRand(n=500)
dep = Classical.InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()

@time poincare_explorer(E, DynSys(), ic_alg)

q0, p0 = initial_conditions(g, E, alg=ic_alg, params=p)
p₀ = [SVector{2}(p0[i, :]) for i ∈ axes(p0, 1)]
q₀ = [SVector{2}(q0[i, :]) for i ∈ axes(q0, 1)]
z0 = [vcat(p₀[i], q₀[i]) for i ∈ axes(q₀, 1)]
prob = ODEProblem(ż, z0[1], 500., p)
sol = solve(prob, Vern9(), abstol=1e-14, reltol=1e-14, maxiters=1e9)

t = Node(0.)
surface_sc = animate_solution(sol, t)
section_sc = θϕ_sections(sol, t, surface_sc.limits[])
vbox(surface_sc, section_sc, sizes=[0.7, 0.3])

Visualizations.animate(t, (0, 40))
