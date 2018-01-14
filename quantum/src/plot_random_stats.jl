#!/usr/bin/env julia
using ArgParse
using Plots, LaTeXStrings
using StatsBase

include("random_data.jl")
include("regions.jl")
include("dataio.jl")

using RandomData, Regions
using DataIO

pgfplots()

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "-n"
            help = "Length of the smallest ensamble"
            arg_type = Int64
            default = 250
        "-f"
            help = "Ensamble size multiplication factor"
            arg_type = Int64
            default = 100
        "-a"
            help = "alpha parameter for the model distribution"
            arg_type = Float64
            default = 0.4
    end
    parsed_args = parse_args(ARGS, arg_settings)
    n = parsed_args["n"]
    f = parsed_args["f"]
    α = parsed_args["a"]

    return n, f, a
end

function main(prefix, dist, f, n, name)
    plt1 = plot(ylabel=L"\gamma_1", legend=:none)
    plt2 = plot(ylabel=L"\kappa", legend=:none)
    for i in 1:10
        sp_regs = regions(rand(dist, f*i*n), f)
        scatter!(plt1, [length(sp_regs[1])], skewness.(sp_regs)')
        scatter!(plt2, [length(sp_regs[1])], kurtosis.(sp_regs)')
    end
    plot!(plt1, x->2)
    plot!(plt1, x->0.631111)
    plot!(plt2, x->6)
    plot!(plt2, x->0.245089)
    savefig(plt1, "$prefix/skewness_dispersion_$name.pdf")
    savefig(plt2, "$prefix/kurtosis_dispersion_$name.pdf")
end

function main()
    n, f, α = input_param()
    dists = [Rayleigh(√2/π), Exponential(), PoissonWigner(α)]
    names = ["wigner", "poisson", "model"]
    prefix = "../output/random_data/"
    for (dist, name) in zip(dists, names)
        main(prefix, dist, f, n, name)
    end
end

main()
