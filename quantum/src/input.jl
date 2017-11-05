#!/usr/bin/env julia
using ArgParse

function ArgParse.parse_item(::Type{Union{Int64,Float64}}, x::AbstractString)
    try
        return parse(Int, x)
    catch ArgumentError
        return parse(Float64, x)
    end
end

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "-b"
            help = "Hamiltonian B parameter"
            arg_type = Float64
            default = [0.55]
            nargs = '+'
        "-d"
            help = "Hamiltonian D parameter"
            arg_type = Float64
            default = 0.4
        "-n"
            help = "Diagonalisation basis size"
            arg_type = Int64
            default = 340
        "--delta_st"
            help = "The maximum energy difference between " *
                "2 diagonalisation bases"
            arg_type = Float64
            default = [1e-9]
            nargs = '+'
        "--epsilon"
            help = "The maximum energy difference between 2 levels " *
                "in the two-dimensional representation"
            arg_type = Float64
            default = 1e-8
        "--bin_size"
            help = "Histogram bin size"
            arg_type = Float64
            default = 0.25
        "--slices", "-s"
            help = "Slices for the irreducible representations"
            arg_type = Int64
            default = 0
        "--idx_slices", "-i"
            help = "Index slices for the irreducible representations"
            arg_type = Int64
            nargs = '+'
            default = [0]
        "--energy_slices", "-e"
            help = "Energy slices for the irreducible representations"
            arg_type = Float64
            nargs = '+'
            default = [0.]
    end
    parsed_args = parse_args(ARGS, arg_settings)
    B = parsed_args["b"]
    D = parsed_args["d"]
    N = parsed_args["n"]
    δ = parsed_args["delta_st"]
    ϵ = parsed_args["epsilon"]
    bin_size = parsed_args["bin_size"]
    slices = parsed_args["slices"]
    idx_slices = parsed_args["idx_slices"]
    energy_slices = parsed_args["energy_slices"]

    options = [slices, idx_slices, energy_slices]
    selection = trues(3)
    selection[1] = (slices != 0)
    selection[2] = (idx_slices != [0])
    selection[3] = (energy_slices != [0.])

    if isempty(options[selection])
        selection[1] = true
        options[1] = 1
    end

    B, D, N, δ, ϵ, bin_size, options[selection][end]
end
