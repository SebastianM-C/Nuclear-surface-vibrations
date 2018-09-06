module DataBaseInterface

export DataBase, compatible, update!, append_with_missing!, update_file, nonnothingtype

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
        columns = Dict{String, Union{Union, Type}}(columns)
        for c in names(df)
            categorical!(df, c)
            df[c] = allowmissing(replace(df[c], missing=>nothing))    # HACK
            columns[string(c)] = eltype(df[c])
        end
        any(.!haskey.(Ref(df), Symbol.(keys(columns)))) && throw(ErrorException("Invalid DataFrame!\n$df"))
        new(location, columns, df)
    end
end

nonnothingtype(::Type{Union{T, Nothing}}) where {T} = T
nonnothingtype(::Type{Union{Missing, CategoricalValue{Union{Nothing, T},R}}}) where {T, R} = T
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
    @debug "Deleting" count(cond)
    deleterows!(db.df, axes(db.df, 1)[cond])
end

function append_with_missing!(db::DataBase, df)
    @debug "Appending" size(df, 1)
    for c in setdiff(names(db.df), names(df))
        df[c] = CategoricalArray{eltype(db.df[c])}(fill(missing, size(df, 1)))
    end
    append!(db.df, df)
end

function fix_column_types(db::DataBase, df::AbstractDataFrame)
    coldiff = setdiff(string.(names(df)), keys(db.columns))
    coldict = Dict(string.(names(df)) .=> eltypes(df))
    # Check if nothings were introduced by append_with_missing!
    to_fix = compatible(db.df[:, names(df)], Dict(names(df) .=> nothing))
    @debug "fix" db.df[:, names(df)] to_fix
    if all(to_fix .=== true)
        for (k,v) in coldict
            db.df[Symbol(k)] = CategoricalArray{v}(fill(missing, length(db.df[Symbol(k)])))
        end
    end
    if !isempty(coldiff)
        @debug "db" db.columns
        push!.(Ref(db.columns), coldict)
        @debug "check" db.columns

        for (k,v) in coldict
            db.df[Symbol(k)] = CategoricalArray{v}(db.df[Symbol(k)])
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
    append_cloned!(df1, df2, .!cond)
end

function append_cloned!(df1, df2, cond)
    if count(cond) > 0
        @debug "Cloning" count(cond) cond
        df_ = DataFrame()
        for c in names(df1)
            df_[c] = df1[c][cond]
        end
        @debug "Updating clone"
        for c in names(df2)
            df_[c][cond] .= df2[c]
        end
        @debug "check clone" colwise(length, df_)
        append!(df1, df_[names(df1)])
        @debug "check update" colwise(length, df1) size(df1)
    end
end

function update_file(db)
    # HACK
    for c in names(db.df)
        db.df[c] = replace(db.df[c], nothing=>missing)
    end
    CSV.write(joinpath(db.location...), db.df)
end

end  # module DataBaseInterface
