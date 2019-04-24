module Reductions

export mean_over_ic, mean_over_E

using ..Parameters
using ..Hamiltonian
using ..DataBaseInterface
using ..InitialConditions
using ..Lyapunov
using ..DInfty
using ..Utils
using ..Classical: AbstractAlgorithm

using Base.Threads
using Plots, LaTeXStrings
using Query, StatsPlots
using StatsBase
using Statistics
using DataFrames
using StorageGraphs
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

function average(df)
    sort!(df, :E)
    average(df[:E], df[:val])
end

function edge_max(v)
    sorted = sort(v)
    Δ = length(v) * diff(sorted) / (sorted[end] - sorted[1])
    sorted[2:end][Δ .< 10][end]
end

function hist_mean(v)
    mean(select_after_first_max(v))
end

function select_after_first_max(v; nbins=50)
    hist = fit(Histogram, v, nbins=nbins, closed=:right)
    firstmax = findlocalmaxima(hist.weights)[1][1]
    v[v .> hist.edges[1][firstmax+1]]
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

function mean_over_ic(g::StorageGraph, s::Symbol, alg, ic_alg::InitialConditionsAlgorithm,
        p::PhysicalParameters, Einterval::Interval=0..Inf;
        reduction=hist_mean)
    pre_dep = (A=p.A,)=>(D=p.D,)=>(B=p.B,)
    E_vals = g[pre_dep, :E]
    filter!(E->E ∈ Einterval, E_vals)
    vals = Vector{typeof(values(alg)[1].T)}(undef, length(E_vals))
    @threads for i in eachindex(E_vals)
        ic_dep = InitialConditions.depchain(p, E_vals[i], ic_alg)
        vals[i] = reduction(g[s, ic_dep..., alg])
    end
    DataFrame(:E=>E_vals, :val=>vals)
end

function mean_over_ic(g::StorageGraph, alg::LyapunovAlgorithm, ic_alg; params=PhysicalParameters(),
        Einterval::Interval=0..Inf, reduction=hist_mean,
        plt=plot(), fnt=font(12, "Times"), width=800, height=600)
    df, t = @timed mean_over_ic(g, :λ, (λ_alg=alg,), ic_alg, params, Einterval, reduction=reduction)
    @info "Averaging took $t seconds."
    df |> @map({_.E, λ=_.val}) |> @orderby(_.E) |>
        @df plot!(plt, :E, :λ, m=2, xlabel=L"E", ylabel=L"\lambda",
            framestyle=:box, legend=false,
            size=(width,height),
            guidefont=fnt, tickfont=fnt)
end

function mean_over_ic(alg::DInftyAlgorithm, ic_alg; params=PhysicalParameters(),
        Einterval::Interval=0..Inf, reduction=hist_mean,
        plt=plot(), fnt=font(12, "Times"), width=800, height=600)
    df = mean_over_ic(:d∞, alg, ic_alg, params, Einterval, reduction=reduction) |>
        @map({_.E, d∞=_.val}) |> @orderby(_.E) |>
        @df plot!(plt, :E, :d∞, m=2, xlabel=L"E", ylabel=L"d_\infty",
            framestyle=:box, legend=false,
            size=(width,height),
            guidefont=fnt, tickfont=fnt)
end

function mean_over_E(s::Symbol, alg, ic_alg::InitialConditionsAlgorithm,
        Einterval::Interval=0..Inf; ic_reduction=hist_mean, reduction=average)
    df = collect_data(alg, ic_alg, Einterval)
    result = by(df, :B, df->DataFrame(v=reduction(by(df, :E,
        df->reduce_col(df, s, ic_reduction))[[:val, :E]])))
    result[:B] = Array(result[:B])

    return result
end

function mean_over_E(alg::LyapunovAlgorithm, Einterval::Interval=0..Inf;
        ic_alg::InitialConditionsAlgorithm,
        ic_reduction=hist_mean, reduction=average,
        plt=plot(), fnt=font(12, "Times"), width=800, height=600, label="")
    mean_over_E(:λs, alg, ic_alg, Einterval;
        ic_reduction=ic_reduction, reduction=reduction) |>
        @map({λ = _.v, _.B}) |> @orderby(_.B) |>
        @df plot!(plt, :B, :λ, m=2, xlabel=L"B", ylabel=L"\lambda", framestyle=:box,
            label=label, size=(width,height), guidefont=fnt, tickfont=fnt)
end

function mean_over_E(alg, Eintervals::NTuple{N, Interval};
        ic_alg::InitialConditionsAlgorithm,
        ic_reduction=hist_mean, reduction=average,
        plt=plot(), fnt=font(12, "Times"), width=800, height=600) where N
    for i=1:N
        plt = mean_over_E(alg, Eintervals[i]; ic_alg=ic_alg, ic_reduction=ic_reduction,
                reduction=reduction, plt=plt, fnt=fnt, width=width, height=height,
                label=L"$E \in "*"$(Eintervals[i])\$")
    end

    return plt
end

function mean_over_E(alg::DInftyAlgorithm, Einterval::Interval=0..Inf;
        ic_alg::InitialConditionsAlgorithm,
        ic_reduction=hist_mean, reduction=average,
        plt=plot(), fnt=font(12, "Times"), width=800, height=600, label="")
    mean_over_E(:d∞, alg, ic_alg, Einterval;
        ic_reduction=ic_reduction, reduction=reduction) |>
        @map({d = _.v, _.B}) |> @orderby(_.B) |>
        @df plot!(plt, :B, :d, m=2, xlabel=L"B", ylabel=L"d_\infty", framestyle=:box,
            label=label, size=(width,height), guidefont=fnt, tickfont=fnt)
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
