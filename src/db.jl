module DataBaseInterface

export DataBase, compatible, update!, append_with_missing!, update_file, nonnothingtype

using DataFrames, CSV
using ..Classical: AbstractAlgorithm
using DiffEqBase: AbstractODEAlgorithm

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

# ==ₘ(a, b::NamedTuple) = a .== string(b)
#
# ==ₘ(a, b::AbstractAlgorithm) = a .== string(typeof(b))
#
# ==ₘ(a, b::AbstractODEAlgorithm) = a .== string(b)
#
# ==ₘ(a, b) = a .== b

# function compatible(df::AbstractDataFrame, alg::AbstractAlgorithm)
#     vals = Dict(f=>getfield(alg, f) for f in fieldnames(typeof(alg)))
#     compatible(df, vals)
# end

# function compatible(df::AbstractDataFrame, vals::AbstractDict)
#     # extend with missing
#     fill_diff!(df, keys(vals))
#     reduce((x,y)->x.&y, [df[k] .== v for (k,v) in vals])
# end

function compatible(df::AbstractDataFrame, vals::AbstractDict, ⊗= ==)
    # extend with missing
    fill_diff!(df, keys(vals))
    reduce((x,y)->x.&y, [Array(df[k]) .⊗ v for (k,v) in vals])
end

function DataFrames.deleterows!(db::DataBase, cond)
    @debug "Deleting rows" count(cond)
    deleterows!(db.df, axes(db.df, 1)[cond])
end

function append_with_missing!(db::DataBase, df)
    @debug "Appending with missing" size(df)
    for c in setdiff(names(db.df), names(df))
        df[c] = CategoricalArray{eltype(db.df[c])}(fill(missing, size(df, 1)))
    end
    append!(db.df, df)
end

function fix_column_types(db::DataBase, df::AbstractDataFrame)
    coldiff = setdiff(string.(names(df)), keys(db.columns))
    coldict = Dict(string.(names(df)) .=> eltypes(df))
    # Check if nothings were introduced by append_with_missing!
    rows_to_fix = compatible(db.df[:, names(df)], Dict(names(df) .=> nothing))
    rows_to_fix = replace(rows_to_fix, missing=>false)  # already ok

    if any(rows_to_fix)
        @debug "Fixing missing values" db.df[:, names(df)] rows_to_fix coldiff
        for c ∈ names(df)
            db.df[rows_to_fix, c] = replace(db.df[rows_to_fix, c], nothing=>missing)
        end
    end

    if !isempty(coldiff)
        for (k,v) in coldict
            db.df[Symbol(k)] = CategoricalArray{v}(db.df[Symbol(k)])
        end
        @debug "Fixed column types"
    end
end

function update!(df1, df2, cond)
    cond = BitArray(replace(cond, missing=>true))
    if count(cond) > 0
        @debug "Updating in-place" count(cond)
        for c in names(df2)
            df1[c][cond] .= df2[c]
        end
    end
    append_cloned!(df1, df2, .!cond)
end

function append_cloned!(df1, df2, cond)
    if count(cond) > 0
        @debug "Cloning" df1, df2, count(cond) cond
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

function update_file(db)
    # HACK
    for c in names(db.df)
        db.df[c] = replace(db.df[c], nothing=>missing)
    end
    CSV.write(joinpath(db.location...), db.df)
end

end  # module DataBaseInterface
