#!/usr/bin/env julia
using ArgParse

include("random_data.jl")
using RandomData

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

function main(prefix, bin_size, slices, N, α)
    plot_dist(prefix, bin_size, slices, N, Rayleigh(√(2/π)), "wigner")
    plot_dist(prefix, bin_size, slices, N, Exponential(), "poisson")
    plot_dist(prefix, bin_size, slices, N, PoissonWigner(α), "model_$α")
end

function main()
    N, α, bin_size, slices = input_param()
    prefix = "../Output/random_data/"
    if !isdir(prefix)
        mkpath(prefix)
    end

    for n in N
        main(prefix, bin_size, slices, n, α)
    end
end

main()
