using Test
using DataFrames, CSV

include("../src/db.jl")
using .DataBaseInterface

global df = DataFrame(:a=>categorical([missing,2]),
    :b=>categorical(allowmissing([2.,3])),
    :c=>categorical(["Alg{Param}",missing]))

const types = [Union{Missing, Int}, Union{Missing, Float64}, Union{Missing, String}]

const location = (".", "test.csv")

@testset "DB creation" begin
    db = DataBase(location, df)
    @test db.df === df
    @test db.location == location
    # Use non-categorical values untill
    # https://github.com/JuliaData/CSV.jl/issues/263
    # is fixed
    @test_skip db.columns == Dict(string.(names(df)) .=> types)
end

@testset "DB IO" begin
    db = DataBase(location, Dict(string.(names(df)) .=> types))
    @test all([all(df[c] .==ₘ db.df[c]) for c in names(df)])
end

@testset "Exported functions" begin
    @test missing ==ₘ missing
    @test (2 ==ₘ missing) == false
    @test (missing ==ₘ 2) == false
    @test 2 ==ₘ 2
    @test (missing ==ₘ "Alg{P}") == false

    # db = DataBase(location, df)
    # filtered_df, cond = compatible(db.df, Dict(:a=>2,:c=>missing))
    # @test count(cond) == 1
    # @test cond[1] == false && cond[2] == true
    # @test all([all(filtered_df[c] .==ₘ db.df[c][2,:]) for c in names(df)])
    # @test isa(filtered_df, SubDataFrame)
    # f2, c2 = compatible(db.df, Dict(:d=>3))
    # @test size(f2, 1) == 0

    DataBaseInterface.deleterows!(db, cond)
    @test size(db.df, 1) == 1

    df_ = DataFrame(:a=>10, :b=>3.14159, :c=>"$(Val(:a))")
    for c in names(df_)
        categorical!(df_, c)
    end
    allowmissing!(df_)
    # update!(db, df_, false, cond)
    # @test size(db.df) == (2, 3)
    # append!(df, df_[names(df)])
    # @test all([all(df[c] .==ₘ db.df[c]) for c in names(df)])

    # df2 = DataFrame(:a=>1, :b=>2., :c=>"$(Val(false))", :d=>2.)
    # allowmissing!(df2)
    # db2 = DataBase((location[1], "test2.csv"), df2)
    # update!(db2, df_, false, [true])
    # rm("test2.csv")
end
