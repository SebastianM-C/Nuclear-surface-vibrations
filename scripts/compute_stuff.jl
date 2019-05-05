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

function compute!(g, Elist, Blist, T)
    times = Float64[]

    @time @showprogress for B in Blist
        @showprogress for E in Elist
            # λ, t = @timed Lyapunov.λmap!(g, E, params=PhysicalParameters(B=B), alg=DynSys(T=T))
            # λ, t = @timed Lyapunov.λmap!(g, E, params=PhysicalParameters(B=B), alg=TimeRescaling(T=T))
            d, t = @timed DInfty.d∞!(g, E, params=PhysicalParameters(B=B), alg=DInftyAlgorithm(T=T))
            push!(times, t)
        end
        savechanges(g)
        if isfile("STOP")
            @info "Stopping now at $B, $E"
            rm("STOP")
            break
        end
    end
    @info "Done"
    return times
end

E = 0.03
ic_alg = PoincareRand(n=500)
p = PhysicalParameters(B=0.5)
ic_dep = Classical.InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()

@profiler Lyapunov.λmap!(g, E, ic_alg=ic_alg, params=PhysicalParameters(B=0.2))

with_logger(dbg) do
    @time Lyapunov.λmap!(g, E, ic_alg=ic_alg, params=PhysicalParameters(B=0.22), alg=DynSys(T=1e5))
end

Bs = setdiff(0.1:0.02:0.6, 0.1:0.1:0.6)
times = compute!(g, 0.01:0.01:10, Bs, 1e5)
using Serialization
times = serialize("/mnt/storage/Nuclear-surface-vibrations/output/classical/times.jls", times)
varinfo()

using Plots
plot(times, ylabel="compute+add to graph time", xlabel="run number", legend=nothing)
savefig("/mnt/storage/Nuclear-surface-vibrations/output/classical/run2_B0.1-0.6_E0.1-10_T1e5_dinf.pdf")


q0, p0 = InitialConditions.initial_conditions!(g, E, alg=ic_alg, params=p)
@time l=λmap(p0, q0, DynSys(T=1e5), params=p)
@time λmap(p0, q0, TimeRescaling(), params=p)

@time l = g[:λ, ic_dep..., (λ_alg=DynSys(T=1e5),)][1]

@time d = d∞(p0, q0, DInftyAlgorithm(T=1e5), params=p)
DInfty.d∞!(g, E, alg=DInftyAlgorithm(T=1e5), ic_alg=ic_alg, params=p)

histogram(d, nbins=50)
histogram(l, nbins=50)

histogram(Γ.(l,d), nbins=50)
