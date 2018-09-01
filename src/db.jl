module DataBaseInterface

export DataBase, ==ₘ, compatible, update!

using DataFrames, CSV

struct DataBase
    location::Tuple{String, String}
    columns::Dict{String, Union}
    df::DataFrame

    function DataBase(location, df::DataFrame)
        columns = Dict(string.(names(df)) .=> eltypes(df))
        CSV.write(joinpath(location...), df)
        new(location, columns, df)
    end
    function DataBase(location, columns::Dict{String, Union})
        # https://github.com/JuliaData/CSV.jl/issues/263
        df = CSV.read(joinpath(location...), use_mmap=!Sys.iswindows(), types=columns)
        for c in names(df)
            categorical!(df, c)
        end
        any(.!haskey.(Ref(df), Symbol.(keys(columns)))) && throw(ErrorException("Invalid DataFrame!\n$df"))
        new(location, columns, df)
    end
end

==ₘ(v, x) = isa(x, Missing) ? isa(v, Missing) : v == x
==ₘ(v, x::Val) = isa(x, Missing) ? isa(v, Missing) : v == "$x"

function compatible(db::DataBase, vals)
    cond = reduce((x,y)->x.&y, [db.df[k] .==ₘ v for (k,v) in vals])
    db.df[cond[.!isa.(cond, Missing)], :], cond
end

function deleterows!(db::DataBase, cond)
    @debug "Deleted" count(cond)
    DataFrames.deleterows!(db.df, axes(db.df, 1)[cond])
end

function update!(db, df, recompute, cond)
    # delete the old values
    if recompute && count(cond) > 0
        @debug "Deleted" count(cond)
        deleterows!(df, cond)
    end
    # add the new values
    for c in setdiff(names(db.df), names(df))
        df[c] = categorical(fill(missing, size(df, 1)))
    end
    append!(db.df, df[names(db.df)])
    @debug "total size" size(db.df)
    CSV.write(joinpath(db.location...), db.df)
end

end  # module DataBaseInterface
