using Distributed

@progress for i in 1:7
    addprocs([("cn$i", 40)], exename="/mnt/storage/julia.sh")
end

addprocs(12)

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
g = initalize()

@profiler Classical.Lyapunov.λmap!(g, E, ic_alg=PoincareRand(n=500), params=PhysicalParameters(B=0.2))

with_logger(dbg) do
    @time Classical.Lyapunov.λmap!(g, E, ic_alg=PoincareRand(n=500), params=PhysicalParameters(B=0.21), alg=DynSys())
end

times = Float64[]
@progress for E in 10.:10.:1000
    λ, t = @timed Classical.Lyapunov.λmap!(g, E, ic_alg=PoincareRand(n=500), params=PhysicalParameters(B=0.55), alg=DynSys())
    push!(times, t)
end
Classical.DataBaseInterface.savechanges(g)

using Plots
plot(times)

using .Visualizations

poincare_explorer(E, DynSys(), PoincareRand(n=10))
