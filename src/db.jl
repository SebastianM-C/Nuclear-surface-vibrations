module DataBaseInterface

export initalize, savechanges

using StorageGraphs
using LightGraphs

function initalize(root=(@__DIR__)*"../output/classical")
    path = root * "/graph.jls"
    if isfile(path)
        return loadgraph(path, SGNativeFormat())
    else
        return StorageGraph()
    end
end

function savechanges(g, root=(@__DIR__)*"../output/classical")
    savegraph(root * "/graph.jls", g, SGNativeFormat())
    savegraph(root * "/graph.bson", g, :g, SGBSONFormat())
end

end  # module DataBaseInterface
