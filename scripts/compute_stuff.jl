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
        if isfile("$(@__DIR__)../STOP")
            @info "Stopping now at $B, $E"
            rm("$(@__DIR__)../STOP")
            break
        end
    end
    @info "Done"
    return times
end

dbg = FilteredLogger(module_filter, ConsoleLogger(stdout, Logging.Debug))

E = 10.
ic_alg=PoincareRand(n=500)
@time g = initialize()

@profiler Classical.Lyapunov.λmap!(g, E, ic_alg=ic_alg, params=PhysicalParameters(B=0.2))

with_logger(dbg) do
    @time Classical.Lyapunov.λmap!(g, E, ic_alg=ic_alg, params=PhysicalParameters(B=0.22), alg=DynSys(T=1e5))
end

times = compute(g, 10:10:3000, 0.55, 1e6)

varinfo()

using Plots
plot(times, ylabel="compute+add to graph time", xlabel="run number", legend=nothing)
savefig("/mnt/storage/Nuclear-surface-vibrations/output/classical/run_B0.1-0.6_E10-3000_T1e6.pdf")
