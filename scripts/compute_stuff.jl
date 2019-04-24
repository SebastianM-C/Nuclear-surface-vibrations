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
using ProgressMeter
include("$(pwd())/scripts/dbg.jl")

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

E = 120.
ic_alg = PoincareRand(n=500)
p = PhysicalParameters(B=0.5)
ic_dep = Classical.InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()

@profiler Classical.Lyapunov.λmap!(g, E, ic_alg=ic_alg, params=PhysicalParameters(B=0.2))

with_logger(dbg) do
    @time Classical.Lyapunov.λmap!(g, E, ic_alg=ic_alg, params=PhysicalParameters(B=0.22), alg=DynSys(T=1e5))
end

times = compute(g, 10:10:3000, 0.55, 1e6)
using Serialization
times = serialize("/mnt/storage/Nuclear-surface-vibrations/output/classical/times.jls")
varinfo()

using Plots
plot(times, ylabel="compute+add to graph time", xlabel="run number", legend=nothing)
savefig("/mnt/storage/Nuclear-surface-vibrations/output/classical/run_B0.55_E10-3000_T1e6.pdf")


q0, p0 = initial_conditions(g, E, alg=ic_alg, params=p)
@time l=λmap(p0, q0, DynSys(T=1e5), params=p)
@time λmap(p0, q0, TimeRescaling(), params=p)

@time l = g[:λ, ic_dep..., (λ_alg=DynSys(T=1e5),)]

@time d = d∞(p0, q0, DInftyAlgorithm(T=1e5), params=p)
histogram(d, nbins=50)
histogram(l, nbins=50)

histogram(Γ.(l,d), nbins=50)
