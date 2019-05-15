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

function select_after_first_max(v; nbins=50, lt = 0.05, ut = 0.1)
    if all(v .< lt)
        return v
    end

    hist = fit(Histogram, v, nbins=nbins, closed=:right)
    firstmax = findlocalmaxima(hist.weights)[1][1]
    if hist.edges[1][firstmax+1] > ut
        return v
    else
        return v[v .> hist.edges[1][firstmax+1]]
    end
end

function select_max_bin(v, t=3)
    hist = fit(Histogram, v, nbins=50, closed=:right)
    threshold = hist.edges[1][t]
    v = v[v.>threshold]
    idx = indmax(hist.weights[t+1:end])
    a = hist.edges[1][t+1:end][idx]
    b = hist.edges[1][t+1:end][idx+1]

    return v[(v.>a) .& (v.<=b)]
end

function hist_max(v, t=3)
    maximum(select_max_bin(v, t))
end

function DInfty.Γ(E, reduction, d0, p)
    λ(E) = reduction(λmap(E, B=p[2], d0=d0))
    d_inf(E) = mean(d∞(E, p, d0))
    Γ(λ(E), d_inf(E))
end

function mean_over_ic(g::StorageGraph, s::Symbol, alg::NamedTuple,
        ic_alg::InitialConditionsAlgorithm,
        p::PhysicalParameters, Einterval=0..Inf;
        reduction=hist_mean)
    pre_dep = (A=p.A,)=>(D=p.D,)=>(B=p.B,)
    E_vals = g[:E, pre_dep]
    filter!(E->E ∈ Einterval, E_vals)
    vals = Vector{typeof(values(alg)[1].T)}(undef, length(E_vals))
    @threads for i in eachindex(E_vals)
        ic_dep = InitialConditions.depchain(p, E_vals[i], ic_alg)
        vals[i] = reduction(g[s, ic_dep..., alg][1])
    end
    DataFrame(:E=>E_vals, :val=>vals)
end

function mean_over_ic(g::StorageGraph, alg::LyapunovAlgorithm, ic_alg;
        params=PhysicalParameters(), Einterval=0..Inf, reduction=hist_mean,
        plt=plot(), kwargs...)
    df, t = @timed mean_over_ic(g, :λ, (λ_alg=alg,), ic_alg, params,
        Einterval, reduction=reduction)
    @debug "Averaging took $t seconds."
    df |> @map({_.E, λ=_.val}) |> @orderby(_.E) |> DataFrame |>
        @df plot!(plt, :E, :λ, m=3, xlabel=L"E", ylabel=L"\lambda",
            legend=false; kwargs...)
end

function mean_over_ic(g::StorageGraph, alg::DInftyAlgorithm, ic_alg;
        params=PhysicalParameters(), Einterval=0..Inf, reduction=hist_mean,
        plt=plot(), kwargs...)
    df, t = @timed mean_over_ic(g, :d∞, (d∞_alg=alg,), ic_alg, params,
        Einterval, reduction=reduction)
    @debug "Averaging took $t seconds."
    df |> @map({_.E, d∞=_.val}) |> @orderby(_.E) |> DataFrame |>
        @df plot!(plt, :E, :d∞, m=3, xlabel=L"E", ylabel=L"d_\infty",
            legend=false; kwargs...)
end

function mean_over_E(g::StorageGraph, s::Symbol, alg::NamedTuple,
        ic_alg::InitialConditionsAlgorithm,
        p::PhysicalParameters, Einterval=0..Inf, Binterval=0..1;
        ic_reduction=hist_mean, reduction=average)
    pre_dep = (A=p.A,)=>(D=p.D,)
    B_vals = g[:B, pre_dep]
    filter!(B->B ∈ Binterval, B_vals)
    vals = Vector{typeof(values(alg)[1].T)}(undef, length(B_vals))
    params = PhysicalParameters(A=p.A, D=p.D, B=B_vals[1])

    for i in eachindex(B_vals)
        @info "B: $(B_vals[i])" progress=i/length(B_vals)
        params = PhysicalParameters(A=p.A, D=p.D, B=B_vals[i])
        vals[i] = reduction(mean_over_ic(g, s, alg, ic_alg, params, Einterval,
            reduction=ic_reduction))
    end

    DataFrame(:B=>B_vals, :val=>vals)
end

function mean_over_E(g::StorageGraph, alg::LyapunovAlgorithm, Einterval=0..Inf;
        A=1, D=0.4, Binterval=0..1, ic_alg::InitialConditionsAlgorithm,
        ic_reduction=hist_mean, reduction=average,
        plt=plot(), kwargs...)
    df = mean_over_E(g, :λ, (λ_alg=alg,), ic_alg, PhysicalParameters(A=A,D=D,B=0.),
        Einterval, Binterval; ic_reduction=ic_reduction, reduction=reduction)
    df |> @map({λ = _.val, _.B}) |> @orderby(_.B) |>
        @df plot!(plt, :B, :λ, m=3, xlabel=L"B", ylabel=L"\lambda"; kwargs...)
end

function mean_over_E(g::StorageGraph, alg::DInftyAlgorithm, Einterval=0..Inf;
        A=1, D=0.4, Binterval=0..1, ic_alg::InitialConditionsAlgorithm,
        ic_reduction=hist_mean, reduction=average,
        plt=plot(), kwargs...)
    df = mean_over_E(:d∞, (d∞_alg=alg,), ic_alg, PhysicalParameters(A=A,D=D,B=0.),
        Einterval, Binterval; ic_reduction=ic_reduction, reduction=reduction)
    df |> @map({d = _.val, _.B}) |> @orderby(_.B) |>
        @df plot!(plt, :B, :d, m=3, xlabel=L"B", ylabel=L"d_\infty"; kwargs...)
end

function mean_over_E(g::StorageGraph, alg, Eintervals::NTuple{N, T};
        A=1, D=0.4, Binterval=0..1, ic_alg::InitialConditionsAlgorithm,
        ic_reduction=hist_mean, reduction=average,
        plt=plot(), kwargs...) where {N, T}
    for i=1:N
        plt = mean_over_E(g, alg, Eintervals[i]; A=A, D=D, Binterval=Binterval,
                ic_alg=ic_alg, ic_reduction=ic_reduction, reduction=reduction,
                plt=plt, label=L"$E \in "*"$(Eintervals[i])\$", kwargs...)
    end

    return plt
end

function mean_over_E(f::Function, values::Tuple{Symbol, Symbol}, B, Einterval=0..Inf;
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

end  # module Reductions
