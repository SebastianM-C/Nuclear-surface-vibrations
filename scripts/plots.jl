using Distributed

addprocs(12)
# addprocs([("headnode:35011", 40)], exename="/mnt/storage/julia.sh",
#     tunnel=true, dir="/mnt/storage/Nuclear-surface-vibrations")

@time using NuclearSurfaceVibrations
using .Classical
using StorageGraphs
using .Visualizations
using Logging, LoggingExtras
using ProgressMeter

function module_filter(level, message, _module, group, id, file, line; kwargs...)
    Base.moduleroot(_module) == NuclearSurfaceVibrations
end

dbg = FilteredLogger(module_filter, ConsoleLogger(stdout, Logging.Debug))

E = 10.
p = PhysicalParameters(B=0.55)
ic_alg = PoincareRand(n=500)
dep = Classical.InitialConditions.depchain(p,E,ic_alg)
g = initialize()
g[dep..., (λ_alg=DynSys(T=1e6),)]
poincare_explorer(E, DynSys(), ic_alg)

with_logger(dbg) do
    λmap(E, alg=DynSys(), ic_alg=PoincareRand(n=500))
end

with_logger(dbg) do
    @time Classical.Lyapunov.λmap!(g, E, ic_alg=PoincareRand(n=50), alg=DynSys())
end

@time g[foldr(=>, dep)]

@time walkdep(g, foldr(=>, dep))

@time paths_through(g, foldr(=>, dep))
