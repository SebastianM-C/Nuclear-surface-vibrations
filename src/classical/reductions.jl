module Reductions

export mean_over_ic, mean_over_E, compatible

using ..Parameters
using ..Hamiltonian
using ..DataBaseInterface
using ..InitialConditions
using ..Lyapunov
using ..DInfty
using ..Utils
using ..Classical: AbstractAlgorithm

using Plots, LaTeXStrings
using Query, StatPlots
using StatsBase
using Statistics
using DataFrames
using ProgressMeter: @showprogress
using IntervalArithmetic
using Images: findlocalmaxima

function average(x, y)
    int = 0
    for i = 1:length(y) - 1
        int += (y[i+1] + y[i]) * (x[i+1] - x[i]) / 2
    end
    int / (x[end] - x[1])
end

average(df) = average(df[:E], df[:val])

function edge_max(v)
    sorted = sort(v)
    Δ = length(v) * diff(sorted) / (sorted[end] - sorted[1])
    sorted[2:end][Δ .< 10][end]
end

function hist_mean(v)
    hist = fit(Histogram, v, nbins=50, closed=:right)
    firstmax = findlocalmaxima(hist.weights)[1][1]
    mean(v[v .> hist.edges[1][firstmax+1]])
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

function DataBaseInterface.compatible(db::DataBase, ic_alg::InitialConditionsAlgorithm,
        alg::LyapunovAlgorithm, Einterval::Interval=0..Inf)
    n, m, border_n = unpack_with_nothing(ic_alg)
    ic_vals = Dict([:n, :m, :initial_cond_alg, :border_n] .=>
                [n, m, string(typeof(ic_alg)), border_n])
    ic_cond = compatible(db.df, ic_vals)
    E_cond = compatible(db.df, Dict(:E => Einterval), ∈)
    @unpack T, Ttr, d0, upper_threshold, lower_threshold, dt, solver, diff_eq_kwargs = alg

    vals = Dict([:lyap_alg, :lyap_T, :lyap_Ttr, :lyap_d0, :lyap_ut,
                :lyap_lt, :lyap_dt, :lyap_integ, :lyap_diffeq_kw] .=>
                [string(typeof(alg)), T, Ttr, d0, upper_threshold,
                lower_threshold, dt, "$solver", "$diff_eq_kwargs"])
    λcond = compatible(db.df, vals)
    cond = ic_cond .& λcond .& E_cond
    cond = replace(cond, missing=>false)

    db.df[cond, :]
end

function DataBaseInterface.compatible(db::DataBase, ic_alg::InitialConditionsAlgorithm,
        alg::DInftyAlgorithm, Einterval::Interval=0..Inf)
    n, m, border_n = unpack_with_nothing(ic_alg)
    ic_vals = Dict([:n, :m, :initial_cond_alg, :border_n] .=>
                [n, m, string(typeof(ic_alg)), border_n])
    ic_cond = compatible(db.df, ic_vals)
    E_cond = compatible(db.df, Dict(:E => Einterval), ∈)
    @unpack T, d0, solver, diff_eq_kwargs = alg

    vals = Dict([:dinf_alg, :dinf_T, :dinf_d0, :dinf_integ, :dinf_diffeq_kw] .=>
                [string(typeof(alg)), T, d0, "$solver", "$diff_eq_kwargs"])
    dcond = compatible(db.df, vals)
    cond = ic_cond .& dcond .& E_cond
    cond = replace(cond, missing=>false)

    db.df[cond, :]
end

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

arr_type(df, col) = nonnothingtype(eltype(df[col]))

function reduce_col(df, col, r)
    DataFrame(val = r(Array{arr_type(df, col)}(df[col])), B = levels(df[:B]))
end

function collect_data(alg, ic_alg, Einterval)
    db = db_concat()
    df = compatible(db, ic_alg, alg, Einterval)

    for c in [:B, :D]
        df[c] = categorical(Array{arr_type(df, c)}(df[c]))
    end
    df[:E] = Array{arr_type(df, :E)}(df[:E])

    return df
end

function mean_over_ic(s::Symbol, alg, ic_alg::InitialConditionsAlgorithm,
        params::PhysicalParameters, Einterval::Interval=0..Inf;
        reduction=hist_mean)
    collect_data(alg, ic_alg, Einterval)

    by(df |> @filter(_.B .== params.B .& _.D .== params.D) |> DataFrame, :E,
        df->reduce_col(df, s, reduction))
end

function mean_over_ic(alg::LyapunovAlgorithm, ic_alg; params=PhysicalParameters(),
        Einterval::Interval=0..Inf, reduction=hist_mean,
        plt=plot(), fnt=font(12, "Times"), width=800, height=600)
    df = mean_over_E(:λs, alg, ic_alg, params, Einterval, reduction=reduction) |>
        @map({_.E, λ=_.val}) |> @orderby(_.E) |>
        @df plot(:E, :λ, m=2, xlabel=L"E", ylabel=L"\lambda",
            framestyle=:box, legend=false,
            size=(width,height),
            guidefont=fnt, tickfont=fnt)
end

function mean_over_ic(alg::DInftyAlgorithm, ic_alg; params=PhysicalParameters(),
        Einterval::Interval=0..Inf, reduction=hist_mean,
        plt=plot(), fnt=font(12, "Times"), width=800, height=600)
    df = mean_over_E(:d∞, alg, ic_alg, params, Einterval, reduction=reduction) |>
        @map({_.E, d∞=_.val}) |> @orderby(_.E) |>
        @df plot(:E, :d∞, m=2, xlabel=L"E", ylabel=L"d_\infty",
            framestyle=:box, legend=false,
            size=(width,height),
            guidefont=fnt, tickfont=fnt)
end

function mean_over_E(s::Symbol, alg, ic_alg::InitialConditionsAlgorithm,
        Einterval::Interval=0..Inf; ic_reduction=hist_mean, reduction=average)
    collect_data(alg, ic_alg, Einterval)
    by(df, :B, df->DataFrame(v=reduction(by(df, :E,
        df->reduce_col(df, s, ic_reduction))[[:val,:E]])))
end

function mean_over_E(alg::LyapunovAlgorithm; ic_alg::InitialConditionsAlgorithm,
        Einterval::Interval=0..Inf, ic_reduction=hist_mean, reduction=average,
        plt=plot(), fnt=font(12, "Times"), width=800, height=600)
    mean_over_B(:λs, alg, ic_alg, Einterval;
        ic_reduction=ic_reduction, reduction=reduction) |>
        @map({λ = _.v, _.B}) |> @orderby(_.B) |>
        @df plot(:B, :λ, m=2, xlabel=L"B", ylabel=L"\lambda", framestyle=:box,
            legend=false, size=(width,height), guidefont=fnt, tickfont=fnt)
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
