#!/usr/bin/env julia
using Plots, StatPlots, LaTeXStrings
using StatsBase
# pyplot()
pgfplots()

include("$(pwd())/input.jl")
include("$(pwd())/regions.jl")
include("$(pwd())/dataio.jl")
include("$(pwd())/statistics.jl")
include("$(pwd())/random_data.jl")

using Regions, DataIO, Statistics

function plot_P(hists, bin_size)
    nbins = Int(4 / bin_size)
    bars = Matrix{Float64}(nbins, size(hists,1))
    for (i,h) in enumerate(hists)
        bars[:,i] = h.weights
    end
    plt = groupedbar(linspace(0 + bin_size / 2, 4 - bin_size / 2, nbins),
        bars, bar_position=:stack, bar_width=bin_size, framestyle=:box,
        ylims=(0., 1.), xlabel=L"$s$", ylabel=L"$P(s)$", label=""
        # label=[L"$\Gamma_a$", L"$\Gamma_s$", L"$\Gamma_b$"]
        );
    return plt
end

function plot_P_fit!(plt::AbstractPlot, α::Number, bin_size)
    f(x) = model(x, α)
    bins = 0:bin_size:4
    plot!(plt, f, label=L"$\alpha = "*@sprintf("%.2f", α)*L"$");
    model_hist = [1 / bin_size * quadgk(f, bins[i], bins[i+1])[1]
                 for i in 1:size(bins, 1) - 1]
    bar!(plt, bins, model_hist, color=:transparent, bar_width=bin_size,
        label="");
end

function plot_P_fit(Γ, bin_size)
    hists = hist_P(Γ, bin_size)
    α = fit_α(hists, bin_size)
    plt = plot_P(hists, bin_size);
    plot_P_fit!(plt, α.param[1], bin_size)
end

function main(prefix, bin_size, slices)
    Γs = read_Γ(prefix)
    Γ_regs = regions(Γs, slices)

    αs = [α.param[1] for α in fit_α(Γ_regs, bin_size)]
    ηs = [η.(Γ_regs_i(Γ_regs, i)) for i=1:length(Γ_regs[1])]
    avg_ηs = η(Γ_regs)
    γ1 = [skewness.(rel_spacing.(Γ_regs_i(Γ_regs, i))) for i=1:length(Γ_regs[1])]
    kurt = [kurtosis.(rel_spacing.(Γ_regs_i(Γ_regs, i))) for i=1:length(Γ_regs[1])]

    f = 100
    # Errors computed from random data
    # The errors for `α` and `η` are given by the standard deviation
    # of the values taken by `α` and `η` on the `f` corresponding
    # ensambles of random values
    rand_regs = RandomData.rand_spacings(Γ_regs, f, αs)
    ε_αs = Vector{eltype(αs)}(length(Γ_regs[1]))
    ε_ηs = Vector{eltype(αs)}(length(Γ_regs[1]))
    for i=1:length(Γ_regs[1])
        ensambles = regions(Γ_regs_i(rand_regs, i), f)
        rand_αs = [α.param[1] for α in RandomData.fit_α(ensambles, bin_size)]
        rand_ηs = RandomData.η(ensambles)
        ε_αs[i] = std(rand_αs)
        ε_ηs[i] = std(rand_ηs)
    end

    add(prefix, Γ_regs, αs, ηs, avg_ηs, ε_αs, ε_ηs, γ1, kurt)

    for i=1:length(Γ_regs[1])
        plt = plot_P_fit(Γ_regs_i(Γ_regs, i), bin_size)
        # savefig(plt, "$prefix/P(s)_$(Γ_regs_idx(Γ_regs, i)).pdf")
        savefig(plt, "$prefix/P(s)_$(Γ_regs_idx(Γ_regs, i)).tex")
    end
end

function main()
    B, D, N, δ, ϵ, bin_size, slices = input_param()

    for δᵢ in δ
        for b in B
            prefix = "../output/B$b D$D N$N/delta_st_$δᵢ epsilon_$ϵ"
            main(prefix, bin_size, slices)
        end
    end
end

main()
