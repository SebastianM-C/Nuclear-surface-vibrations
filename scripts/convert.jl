using NuclearSurfaceVibrations
using .Classical
using .InitialConditions: initial_conditions!, extract_ics, depchain
using StorageGraphs
using LightGraphs
using Test
using ProgressMeter
using Serialization
include("$(pwd())/scripts/dbg.jl")

g = loadgraph("output/classical/graph.jls.backup2", SGNativeFormat())

function convert_g(g)
    Bs = g[:B]
    h = StorageGraph()

    @showprogress "B" for B in Bs
        Es = g[(A=1,)=>(D=0.4,)=>(B=B,), :E]
        @showprogress "E" for E in Es
            ic_algs = g[(A=1,)=>(D=0.4,)=>(B=B,)=>(E=E,), :ic_alg]
            @showprogress for ic_alg in ic_algs
                p = PhysicalParameters(A=1,D=0.4,B=B)
                ic_nodes = initial_conditions!(g, E, alg=ic_alg, params=p)
                q0, p0 = extract_ics(ic_nodes, ic_alg)
                ic_vertices = get.(Ref(g.index), ic_nodes, 0)
                ic_deps = depchain(p, E, ic_alg)
                all_algs = g[:λ_alg]
                λ_algs = Int[]
                for v in ic_vertices
                    union!(λ_algs, outneighbors(g, v))
                    length(λ_algs) == length(all_algs) && break
                end
                for λ_alg in map(a->g[a], λ_algs)
                    λ = g[:λ, ic_deps..., λ_alg]
                    add_nodes!(h, foldr(=>,(ic_deps..., (q0=q0,p0=p0), λ_alg, (λ=λ,))))
                end
            end
        end
    end
    return h
end

h = convert_g(g)
# serialize("Nuclear-surface-vibrations/output/classical/new_graph.jls", h)
h = deserialize("output/classical/graph.jls")

@test length(g[:λ]) == sum(length.(h[:λ]))
g[:ic_alg]
h[:ic_alg]


B=0.55
E=10
ic_algs = g[:ic_alg, (A=1,)=>(D=0.4,)=>(B=B,)=>(E=E,)]
ic_alg=ic_algs[1]
p = PhysicalParameters(A=1,D=0.4,B=B)

ic_dep = Classical.InitialConditions.depchain(p, E, ic_alg)

@profiler walkdep(g, foldr(=>, ic_dep))
@profiler walkdep(h, foldr(=>, ic_dep))

with_logger(dbg) do
    @time Lyapunov.λmap!(h, E, ic_alg=PoincareRand(n=3), params=p, alg=DynSys())
end

@time g[foldr(=>,ic_dep)]

@profiler Lyapunov.λmap!(h, E, ic_alg=PoincareRand(n=3), params=p, alg=DynSys())

ic_node = InitialConditions.initial_conditions!(h, 21, alg=ic_alg, params=p)
outneighbors(h, h[ic_node])

alg=DynSys()
@profiler g[:λ, foldr(=>, (ic_dep..., (λ_alg=alg,)))]
@profiler g[:λ, ic_dep..., (λ_alg=alg,)]


ne(g)/ne(h)
g.maxid[]/h.maxid[]

@profiler h[h[foldr(=>,ic_dep)][1]]
@time g[foldr(=>,ic_dep)]

@time h[:λ, ic_dep..., (λ_alg=DynSys(),)][1]
@time g[:λ, ic_dep..., (λ_alg=DynSys(),)]

@time h[:λ, foldr(=>, (ic_dep..., (λ_alg=alg,)))][1]
@time g[:λ, foldr(=>, (ic_dep..., (λ_alg=alg,)))]
@time g[ic_dep...]

initial_conditions(g, E, alg=ic_alg, params=p)

varinfo()

savechanges(h)
