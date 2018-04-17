#!/usr/bin/env julia
using ArgParse
using Plots, StatPlots, LaTeXStrings

include("random_data.jl")
using RandomData

pgfplots()

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "-n"
            help = "Number of random spacings"
            arg_type = Int64
            default = [10000]
            nargs = '+'
        "-a", "--alpha"
            help = "Linear superposition coefficient"
            arg_type = Float64
            default = 0.4
        "-b", "--bin_size"
            help = "Bin size"
            arg_type = Float64
            default = 0.25
        "-s", "--slices"
            help = "Number of slices"
            arg_type = Int64
            default = 3
    end
    parsed_args = parse_args(ARGS, arg_settings)
    N = parsed_args["n"]
    α = parsed_args["alpha"]
    bin_size = parsed_args["bin_size"]
    slices = parsed_args["slices"]

    return N, α, bin_size, slices
end

function plot_P(bars, bin_size, dist::Distribution)
    label = "Random data: "*
        L"$\alpha = "*"$(fit_α(bars, bin_size).param[1]),"*
        L"\,\eta\ = "*"$(η(bars))"*L"$"
    plt = histogram(bins=collect(0:bin_size:4),
        bars, framestyle=:box, normed=true, xlims=(0., 4.), ylims=(0., 1.),
        xlabel=L"$s$", ylabel=L"$P(s)$", label=label);
    if typeof(dist) <: PoissonWigner
        plot!(plt, linspace(0, 4, 100), pdf.(PoissonWigner(dist.α),
            linspace(0, 4, 100)), label="Probability distribution function")
    else
        plot!(plt, dist, label="Probability distribution function")
    end
    return plt
end

function _plot_dist(bin_size, distribution, spacings, fname)
    plt = plot_P(spacings, bin_size, distribution)
    savefig(plt, fname)
end

function plot_dist(prefix, bin_size, slices, N, distribution, name)
    spacings = rand(distribution, N)
    sp_regions = regions(spacings, slices)
    _plot_dist(bin_size, distribution, spacings, "$prefix/"*name*"_$N.pdf")

    for i in 1:slices
        _plot_dist(bin_size, distribution, sp_regions[i],
            "$prefix/"*name*"_$N-slice$i.pdf")
    end
end


function main(prefix, bin_size, slices, N, α)
    plot_dist(prefix, bin_size, slices, N, Rayleigh(√(2/π)), "wigner")
    plot_dist(prefix, bin_size, slices, N, Exponential(), "poisson")
    plot_dist(prefix, bin_size, slices, N, PoissonWigner(α), "model_$α")
end

function main()
    N, α, bin_size, slices = input_param()
    prefix = "../output/random_data/"
    if !isdir(prefix)
        mkpath(prefix)
    end

    for n in N
        main(prefix, bin_size, slices, n, α)
    end
end

main()
