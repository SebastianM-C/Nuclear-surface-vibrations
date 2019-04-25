using NuclearSurfaceVibrations
using .Classical
using .InitialConditions: initial_conditions!, extract_ics, depchain
using StorageGraphs
using LightGraphs
using Test
using ProgressMeter

g = loadgraph("output/classical/graph.jls", SGNativeFormat())

function convert_g(g)
    Bs = g[:B]
    h = StorageGraph()

    @showprogress "B $B" for B in Bs
        Es = g[(A=1,)=>(D=0.4,)=>(B=B,), :E]
        @showprogress "E $E" for E in Es
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

@test length(g[:λ]) == sum(length.(h[:λ]))
g[:ic_alg]
h[:ic_alg]

g[:λ_alg]
h[:λ_alg]


B=0.21
E=10
ic_algs = g[(A=1,)=>(D=0.4,)=>(B=B,)=>(E=E,), :ic_alg]
ic_alg=ic_algs[1]
p = PhysicalParameters(A=1,D=0.4,B=B)

ic_dep = Classical.InitialConditions.depchain(p, E, ic_alg)
v = g[foldr(=>,ic_dep)]
g[v[1]]
v = g[(A=1,)=>(D=0.4,)=>(B=0.21,)=>(E=10,), :ic_alg]
g[v[2]]

h[foldr(=>,ic_dep), :q0]
h[(A=1,)=>(D=0.4,)=>(B=0.21,)=>(E=10,), :ic_alg]


h[4]



g[(ic_alg=g[:ic_alg][2],)]
paths_through(g, 2)
outneighbors(g,2)
outneighbors(g,6)


h[:ic_alg]

initial_conditions(g, E, alg=ic_alg, params=p)

varinfo()
