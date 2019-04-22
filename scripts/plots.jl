using Distributed

addprocs(12)
addprocs([("headnode:35011", 40)], exename="/mnt/storage/julia.sh",
    tunnel=true, dir="/mnt/storage/Nuclear-surface-vibrations")

@time using NuclearSurfaceVibrations
using .Classical
using StorageGraphs
using .Visualizations
using Logging, LoggingExtras
using ProgressMeter

function module_filter((level, _module, group, id))
    Base.moduleroot(_module) == NuclearSurfaceVibrations
end

dbg = EarlyFilteredLogger(module_filter, ConsoleLogger(stdout, Logging.Debug))

E = 50.
p = PhysicalParameters(B=0.55)
ic_alg = PoincareRand(n=500)
dep = Classical.InitialConditions.depchain(p,E,ic_alg)
@time g = initialize()
g[dep..., (λ_alg=DynSys(),)]

with_logger(dbg) do
    @time poincare_explorer(E, DynSys(), ic_alg)
end

with_logger(dbg) do
    @time Classical.Lyapunov.λmap!(g, E, ic_alg=ic_alg, alg=DynSys())
end
savechanges(g)
@time g[foldr(=>, dep), :q₂]

@time walkdep(g, foldr(=>, dep))

with_logger(dbg) do

end

@time v = mean_over_ic(g, :λ, (λ_alg=DynSys(),), ic_alg, p)
using Plots
plot(g[(A=p.A,)=>(D=p.D,)=>(B=p.B,), :E], v)
