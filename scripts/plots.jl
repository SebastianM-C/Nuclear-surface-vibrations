using Distributed

addprocs(12)
addprocs([("headnode:35011", 40)], exename="/mnt/storage/julia.sh",
    tunnel=true, dir="/mnt/storage/Nuclear-surface-vibrations")

@time using NuclearSurfaceVibrations
using .Classical
using StorageGraphs
using .Visualizations

E = 50.
p = PhysicalParameters(B=0.55)
ic_alg = PoincareRand(n=500)
dep = Classical.InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()

@time poincare_explorer(E, DynSys(), ic_alg)
