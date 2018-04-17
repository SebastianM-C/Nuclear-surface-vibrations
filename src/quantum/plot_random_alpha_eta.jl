#!/usr/bin/env julia
using ArgParse
using Plots, LaTeXStrings

include("random_data.jl")
include("regions.jl")
using RandomData, Regions

pgfplots()

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "-n"
            help = "Number of random spacings"
            arg_type = Int64
            default = [10000]
            nargs = '+'
        "-m", "--alpha_step"
            help = "Linear superposition coefficient steps"
            arg_type = Int64
            default = 11
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
    m = parsed_args["alpha_step"]
    bin_size = parsed_args["bin_size"]
    slices = parsed_args["slices"]

    return N, m, bin_size, slices
end

function main(target_αs, N, bin_size, slices, name)
    αs, ηs = α_range(target_αs, N, bin_size, slices)

    plt = scatter(ηs[:,1], αs[:,1], aspect_ratio=1, framestyle=:box,
        xlabel=L"\eta", ylabel=L"\alpha", legend=:none)
    savefig(plt, "$name.pdf")

    labels = "slice no.~".*["$i" for i ∈ 1:slices]
    plt = scatter(ηs[:,2:end], αs[:,2:end], aspect_ratio=1, framestyle=:box,
        xlabel=L"\eta", ylabel=L"\alpha", label=labels, legend=:topleft)
    savefig(plt, "$name-$slices-slices.pdf")
end

function main()
    N, m, bin_size, slices = input_param()
    target_αs = linspace(0, 1, m)
    prefix = "../output/random_data/"
    for n in N
        name = "$prefix/alpha(eta)_N$n-$m-points"
        main(target_αs, n, bin_size, slices, name)
    end
end

main()
