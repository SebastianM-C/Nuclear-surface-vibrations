using Distributed

@progress for i in 2:8
    addprocs([("cn$i", 40)], exename="/mnt/storage/julia.sh")
end

addprocs(12)
addprocs([("headnode:35011", 40)], exename="/mnt/storage/julia.sh",
    tunnel=true, dir="/mnt/storage/Nuclear-surface-vibrations")

@time using NuclearSurfaceVibrations
using .Classical
using StorageGraphs
using LightGraphs
using Logging, LoggingExtras
using ProgressMeter

function module_filter(level, message, _module, group, id, file, line; kwargs...)
    Base.moduleroot(_module) == NuclearSurfaceVibrations
end

function compute(g, Elist, Blist, T)
    times = Float64[]

    @time @showprogress for B in Blist
        @showprogress for E in Elist
            λ, t = @timed Classical.Lyapunov.λmap!(g, E, params=PhysicalParameters(B=B), alg=DynSys(T=T))
            push!(times, t)
        end
        savechanges(g)
    end
    @info "Done"
    @time savechanges(g, backup=true)
    return times
end

dbg = FilteredLogger(module_filter, ConsoleLogger(stdout, Logging.Debug))

E = 10.
g = initialize()

@profiler Classical.Lyapunov.λmap!(g, E, ic_alg=PoincareRand(n=500), params=PhysicalParameters(B=0.2))

with_logger(dbg) do
    @time Classical.Lyapunov.λmap!(g, E, ic_alg=PoincareRand(n=500), params=PhysicalParameters(B=0.21), alg=DynSys())
end

times = compute(g, 10:10:3000, (0.55), 1e4)

using Plots
plot(times)

using .Visualizations

poincare_explorer(E, DynSys(), PoincareRand(n=10))
