module Reductions

export mean_as_function_of_B

using ..Parameters: @showprogress
using ..DataBaseInterface
using ..InitialConditions
using ..Lyapunov
using ..DInfty
using ..Utils
using ..Classical: AbstractAlgorithm

using Plots, LaTeXStrings
using Query, StatPlots
using StatsBase
using DataFrames
using ProgressMeter
using IntervalArithmetic

function average(x, y)
    int = 0
    for i = 1:length(y) - 1
        int += (y[i+1] + y[i]) * (x[i+1] - x[i]) / 2
    end
    int / (x[end] - x[1])
end

function edge_max(v)
    sorted = sort(v)
    Δ = length(v) * diff(sorted) / (sorted[end] - sorted[1])
    sorted[2:end][Δ .< 10][end]
end

function hist_avg(v, t=3)
    hist = fit(Histogram, v, nbins=50, closed=:right)
    threshold = hist.edges[1][t]
    mean(v[v.>threshold])
end

function hist_max(v, t=3)
    hist = fit(Histogram, v, nbins=50, closed=:right)
    threshold = hist.edges[1][t]
    v = v[v.>threshold]
    idx = indmax(hist.weights[t+1:end])
    a = hist.edges[1][t+1:end][idx]
    b = hist.edges[1][t+1:end][idx+1]
    maximum(v[(v.>a) .& (v.<=b)])
end

function DInfty.Γ(E, reduction, d0, p)
    λ(E) = reduction(λmap(E, B=p[2], d0=d0))
    d_inf(E) = mean(d∞(E, p, d0))
    Γ(λ(E), d_inf(E))
end

function DataBaseInterface.compatible(db::DataBase, ic_alg::AbstractAlgorithm, alg::AbstractAlgorithm)
    n, m, border_n = unpack_with_nothing(ic_alg)
    ic_vals = Dict([:n, :m, :initial_cond_alg, :border_n] .=>
                [n, m, string(typeof(ic_alg)), border_n])
    ic_cond = compatible(db.df, ic_vals)


function λlist(Elist, Blist=0.55, Dlist=0.4; T=12000., Ttr=5000., recompute=false)
    for D in Dlist
        @showprogress "B" for i = 1:length(Blist)
            @showprogress "λs" for j = 1:length(Elist)
                λs = λmap(Elist[j], B=Blist[i], D=D, T=T, Ttr=Ttr,
                    recompute=recompute)
                if !any(occursin.(r"poincare_lyapunov.*pdf", readdir(prefix))) || recompute
                    coloredpoincare(Elist[j], λs, name="lyapunov", B=Blist[i], D=D)
                end
            end
        end
    end
end

function mean_over_E(f::Function, values::Tuple{Symbol, Symbol}, B, Einterval::Interval=0..Inf;
        reductions=(edge_max,mean))
    dfs = [concat(r"z0.csv", location="classical/B$(B[1])-D0.4",
        re=r"E[0-9]+\.[0-9]+", filter=[:E, v]) |>
        @filter(_.E ∈ Einterval && _.E % 10 .== 0) |> DataFrame for v in values]
    df_vs = [by(dfs[i], :E, df->DataFrame(val = reductions[i](df[values[i]]))) |>
        @orderby(_.E) |> DataFrame for i in eachindex(dfs)]
    df_f = @join(df_vs[1], df_vs[2], _.E, _.E, {val = f(_.val, __.val)}) |> DataFrame
    df_f[:E] = df_vs[1][:E]
    return df_f
end

"""
    mean_as_function_of_B(value::Symbol, B, Einterval::Interval=0..Inf;
           reduction=ch_max, plt=plot(), ylabel="", fnt=font(12, "Times"),
           width=800, height=600, filename="val(E)")

Compute the mean of a stored `value` as a funciton of `B` considering the
given energy interval `Einterval`. Return the corresponding `DataFrame`
"""
function mean_as_function_of_B(value::Symbol, B, Einterval::Interval=0..Inf;
        reduction=edge_max, plt=plot(), ylabel="", fnt=font(12, "Times"),
        width=800, height=600, filename="val(E)")
    df = concat(r"z0.csv", location="classical/B$(B[1])-D0.4",
        re=r"E[0-9]+\.[0-9]+", filter=[:E, value]) |> @filter(_.E ∈ Einterval) |> DataFrame
    df_v = by(df, :E, df->DataFrame(val = reduction(df[value]))) |>
        @orderby(_.E) |> DataFrame
    df_v[:B] = fill(B[1], size(df_v, 1))

    df_v |> @df plot(:E, :val, m=2, xlabel=L"E", ylabel=ylabel,
        framestyle=:box, legend=false,
        size=(width,height),
        guidefont=fnt, tickfont=fnt)
    savefig("../../output/classical/B$(B[1])-D0.4/$filename.pdf")

    for i in 2:length(B)
        df = concat(r"z0.csv", location="classical/B$(B[i])-D0.4",
            re=r"E[0-9]+\.[0-9]+", filter=[:E, value]) |> @filter(_.E ∈ Einterval) |> DataFrame
        df_ = by(df, :E, df->DataFrame(val = reduction(df[value]))) |>
            @orderby(_.E) |> DataFrame
        df_ |> @df plot(:E, :val, m=2, xlabel=L"E", ylabel=ylabel,
            framestyle=:box, legend=false,
            size=(width,height),
            guidefont=fnt, tickfont=fnt)
        savefig("../../output/classical/B$(B[i])-D0.4/$filename.pdf")
        df_[:B] = fill(B[i], size(df_, 1))
        append!(df_v, df_[names(df_v)])
    end

    by(df_v |> @filter(_.E ∈ Einterval) |> DataFrame, :B,
        df->DataFrame(v = average(df[:E], df[:val]))) |>
            @orderby(_.B) |> DataFrame
end

function mean_as_function_of_B(value::Symbol, B, Eintervals::NTuple{N, Interval};
        reduction=ch_max, plt=plot(), ylabel="", fnt=font(12, "Times"),
        width=800, height=600, filename="val(E)") where N
    for i=1:N
        df = mean_as_function_of_B(value, B, Eintervals[i], reduction=reduction,
            plt=plt, ylabel=ylabel, fnt=fnt, width=width, height=height,
            filename=filename)
        plt = df |> @df plot!(plt, :B, :v, m=2, xlabel=L"B", ylabel=ylabel,
            label=L"$E \in "*"$(Eintervals[i])\$", framestyle=:box,
            size=(width,height),
            guidefont=fnt, tickfont=fnt)
    end
    plt
end

end  # module Reductions
