using Distributed

@progress for i in 1:5
    addprocs([("cn$i", 40)], exename="/mnt/storage/julia.sh")
end

# addprocs(20)

@time using NuclearSurfaceVibrations
using .Classical
using StorageGraphs
using LightGraphs
using Logging, LoggingExtras

function module_filter(level, message, _module, group, id, file, line; kwargs...)
    Base.moduleroot(_module) == NuclearSurfaceVibrations
end
dbg = FilteredLogger(module_filter, ConsoleLogger(stdout, Logging.Debug))

E = 10.
q0, p0 = initial_conditions(E, alg=PoincareRand(n=10))
g = Classical.DataBaseInterface.initalize()

@profiler λmap(E, ic_alg=PoincareRand(n=100), params=PhysicalParameters(B=0.55))

with_logger(dbg) do
    @time λmap(E, ic_alg=PoincareRand(n=100), params=PhysicalParameters(B=0.2), alg=DynSys())
end

@progress for E in 10.:10.:1e3
    λmap(E, ic_alg=PoincareRand(n=500), params=PhysicalParameters(B=0.55), alg=DynSys())
end

using .Visualizations

poincare_explorer(E, DynSys(), PoincareRand(n=10))
