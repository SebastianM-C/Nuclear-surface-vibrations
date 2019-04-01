using Distributed
addprocs(20)
using NuclearSurfaceVibrations
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

@profiler λmap(E, ic_alg=PoincareRand(n=7), params=PhysicalParameters(B=0.20))

with_logger(dbg) do
    λmap(E, ic_alg=PoincareRand(n=100), params=PhysicalParameters(B=0.2), alg=DynSys())
end




using .Visualizations

poincare_explorer(E, DynSys(), PoincareRand(n=10))
