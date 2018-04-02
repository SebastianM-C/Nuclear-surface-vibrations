#!/usr/bin/env julia
using ArgParse
using Plots, LaTeXStrings

include("random_data.jl")
include("regions.jl")

using RandomData
using Regions

pgfplots()
# plotlyjs()

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "-f"
            help = "Number of spacings in reference relative to the base sample"
            arg_type = Int64
            default = 100
        "-a", "--alpha"
            help = "Linear superposition coefficient steps"
            arg_type = Float64
            default = 0.4
        "--bin_size"
            help = "Bin size"
            arg_type = Float64
            default = 0.25
        "-b", "--base_n"
            help = "Number of spacings for the smallest sample"
            arg_type = Int64
            default = 250
    end
    parsed_args = parse_args(ARGS, arg_settings)
    f = parsed_args["f"]
    α = parsed_args["alpha"]
    bin_size = parsed_args["bin_size"]
    base_n = parsed_args["base_n"]

    return f, α, bin_size, base_n
end


function main()
    f, target_α, bin_size, base_n = input_param()
    prefix = "../output/random_data/"

    plt1 = plot(ylabel=L"$\eta$", ylims=(0.,1.), framestyle=:box)
    plt2 = plot(ylabel=L"$\alpha$", ylims=(0.,1.), framestyle=:box)

    spacings = rand(PoissonWigner(target_α), f * base_n)
    for i in [1, 2, 4, 5, 10]
        αs = Vector{typeof(target_α)}()
        ηs = zeros(αs)
        sp_regs = regions(spacings, Int(f / i))
        for s in sp_regs
            αᵢ = fit_α(s, bin_size).param[1]
            ηᵢ = η(s)
            push!(αs, αᵢ)
            push!(ηs, ηᵢ)
        end
        scatter!(plt1, [i * base_n], ηs', label="")
        scatter!(plt2, [i * base_n], αs', label="")
    end

    savefig(plt1, "$prefix/var_eta.pdf")
    savefig(plt2, "$prefix/var_alpha.pdf")
end

main()
