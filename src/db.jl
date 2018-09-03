module DataBaseInterface

export DataBase, ==ₘ, compatible, update!, append_with_missing!

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
        # TODO: Switch to JLD2 after https://github.com/simonster/JLD2.jl/issues/101 is fixed
        # https://github.com/JuliaData/CSV.jl/issues/263
        df = CSV.read(joinpath(location...), use_mmap=!Sys.iswindows(), types=columns)
        for c in names(df)
            categorical!(df, c)
        end
        any(.!haskey.(Ref(df), Symbol.(keys(columns)))) && throw(ErrorException("Invalid DataFrame!\n$df"))
        new(location, columns, df)
    end
end

==ₘ(v, x) = isa(x, Missing) ? isa(v, Missing) : isa(v, Missing) ? false : v == x

function fill_diff!(df::AbstractDataFrame, cols)
    for c in setdiff(cols, names(df))
        df[c] = fill(missing, size(df, 1))
    end
end

function compatible(df::AbstractDataFrame, vals)
    # extend with missing
    fill_diff!(df, keys(vals))
    cond = reduce((x,y)->x.&y, [df[k] .==ₘ v for (k,v) in vals])
    subdf = view(df, cond)

    return cond
end

function DataFrames.deleterows!(db::DataBase, cond)
    @debug "Deleted" count(cond)
    deleterows!(db.df, axes(db.df, 1)[cond])
end

function append_with_missing!(db::DataBase, df)
    @debug "Appended" size(df, 1)
    fill_diff!(df, names(db.df))
    append!(db.df, df[names(db.df)])
    CSV.write(joinpath(db.location...), db.df)
end

function update!(db, df, cond)
    subdf = view(db.df, cond)
    if size(subdf, 1) > 0
        for c in names(df)
            subdf[c] .= df[c]
        end
    else
        fill_diff!(db.df, names(df))
        append!(db.df, df[names(db.df)])
    end
    @debug "total size" size(db.df)
    CSV.write(joinpath(db.location...), db.df)
end

end  # module DataBaseInterface
