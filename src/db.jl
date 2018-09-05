module DataBaseInterface

export DataBase, compatible, update!, append_with_missing!, update_file!, nonnothingtype

using DataFrames, CSV

struct DataBase
    location::Tuple{String, String}
    columns::Dict{String, Union}
    df::DataFrame

    function DataBase(location, df::DataFrame)
        columns = Dict(string.(names(df)) .=> eltypes(df))
        # HACK
        for c in names(df)
            df[c] = replace(df[c], nothing=>missing)
        end
        CSV.write(joinpath(location...), df)
        new(location, columns, df)
    end
    function DataBase(location, columns::Dict{String, Union})
        # TODO: Switch to JLD2 after https://github.com/simonster/JLD2.jl/issues/101 is fixed
        # https://github.com/JuliaData/CSV.jl/issues/263
        df = CSV.read(joinpath(location...), use_mmap=!Sys.iswindows(), types=columns)
        for c in names(df)
            df[c] = replace(df[c], missing=>nothing)    # HACK
            categorical!(df, c)
        end
        any(.!haskey.(Ref(df), Symbol.(keys(columns)))) && throw(ErrorException("Invalid DataFrame!\n$df"))
        new(location, columns, df)
    end
end

nonnothingtype(::Type{Union{T, Nothing}}) where {T} = T
nonnothingtype(::Type{Nothing}) = Union{}
nonnothingtype(::Type{T}) where {T} = T
nonnothingtype(::Type{Any}) = Any

function fill_diff!(df::AbstractDataFrame, cols)
    for c in setdiff(cols, names(df))
        df[c] = fill(missing, size(df, 1))
    end
end

function compatible(df::AbstractDataFrame, vals)
    # extend with missing
    fill_diff!(df, keys(vals))
    reduce((x,y)->x.&y, [df[k] .== v for (k,v) in vals])
end

function DataFrames.deleterows!(db::DataBase, cond)
    @debug "Deleted" count(cond)
    deleterows!(db.df, axes(db.df, 1)[cond])
end

function append_with_missing!(db::DataBase, df)
    @debug "Appended" size(df, 1)
    fill_diff!(df, names(db.df))
    append!(db.df, df[names(db.df)])
end

function fix_column_types(db::DataBase, df)
    coldiff = setdiff(Dict(string.(names(df)) .=> eltypes(df)), db.columns)
    @debug "cols" coldiff
    if !isempty(coldiff)
        push!.(Ref(db.columns), coldiff)
        @debug "check" db.columns
        for (k,v) in coldiff
            db.df[Symbol(k)] = Array{v}(db.df[Symbol(k)])
        end
    end
end

function update!(df1, df2, cond)
    cond = BitArray(replace(cond, missing=>true))
    @debug "Updated in-place" count(cond) cond
    if count(cond) > 0
        for c in names(df2)
            df1[c][cond] .= df2[c]
        end
    end
    @debug "done"
    append_cloned!(df1, df2, .!cond)
end

function append_cloned!(df1, df2, cond)
    if count(cond) > 0
        @debug "Cloned" count(cond) cond
        df_ = DataFrame()
        for c in names(df1)
            df_[c] = df1[c][cond]
        end
        for c in names(df2)
            df_[c][cond] .= df2[c]
        end
        append!(df1, df_[names(df1)])
    end
end

function update_file!(db)
    # HACK
    for c in names(db.df)
        db.df[c] = replace(db.df[c], nothing=>missing)
    end
    CSV.write(joinpath(db.location...), db.df)
end

end  # module DataBaseInterface
