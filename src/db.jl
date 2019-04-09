module DataBaseInterface

export initialize, savechanges

using StorageGraphs
using LightGraphs

function initialize(root=(@__DIR__)*"/../output/classical")
    path = root * "/graph.jls"
    if isfile(path)
        try
            return loadgraph(path, SGNativeFormat())
        catch e
            @warn "Got an exception trying to load files" e
            try
                @debug "Loading from BSON backup"
                return loadgraph(root * "/graph.bson", SGBSONFormat())
            catch e
                @debug "Couldn't restore any saved data" e
                return StorageGraph()
            end
        end
    elseif isfile(root * "/graph.bson")
        @info ".jls file not found, trying backup"
        try
            @debug "Loading from BSON backup"
            return loadgraph(root * "/graph.bson", SGBSONFormat())
        catch e
            @debug "Couldn't restore any saved data" e
                return StorageGraph()
        end
    else
        @debug "Couldn't find any saved data"
        return StorageGraph()
    end
end

function savechanges(g, root=(@__DIR__)*"/../output/classical"; backup=false)
    if !ispath(root)
        mkpath(root)
    end
    savegraph(root * "/graph.jls", g, SGNativeFormat())
    if backup
        savegraph(root * "/graph.bson", g, :g, SGBSONFormat())
    end
end

end  # module DataBaseInterface
