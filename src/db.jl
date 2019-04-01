module DataBaseInterface

export initalize, savechanges

using StorageGraphs
using LightGraphs

function initalize()
    path = (@__DIR__) * "/../output/classical/graph.jls"
    if isfile(path)
        return loadgraph(path, SGNativeFormat())
    else
        return StorageGraph()
    end
end

function savechanges(g)
    path = (@__DIR__) * "/../output/classical/"
    savegraph(path * "graph.jls", g, SGNativeFormat())
    # savegraph(path * "graph.jld", g, "g", SGJLDFormat())
end

end  # module DataBaseInterface
