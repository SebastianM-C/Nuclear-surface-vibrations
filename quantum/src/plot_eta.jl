#!/usr/bin/env julia
using ArgParse
using Plots, LaTeXStrings
pgfplots()

include("dataio.jl")
include("regions.jl")
include("statistics.jl")
using DataIO, Regions, Statistics

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
        "--delta_E", "-e"
            help = "Energy slices for the irreducible representations"
            arg_type = Float64
            default = 10.
    end
    parsed_args = parse_args(ARGS, arg_settings)
    B = parsed_args["b"]
    D = parsed_args["d"]
    N = parsed_args["n"]
    δ = parsed_args["delta_st"]
    ϵ = parsed_args["epsilon"]
    energy_slices = parsed_args["delta_E"]

    B, D, N, δ, ϵ, energy_slices
end

function main(prefix, ΔE)
    Γs = read_Γ(prefix)
    Γ_regs = cumulative_regions(Γs, ΔE)
    fix_regs(Γ_regs)
    ηs = Array{eltype(Γs[1])}(length(Γ_regs[1]), length(Γ_regs))
    for i=1:length(Γ_regs[1])
        ηs[i, :] .= η.(Γ_regs_i(Γ_regs, i))
    end

    E_max = ceil(minimum(Γs[i][end] for i in 1:length(Γs)) / ΔE) * ΔE
    dΓ = regions(Γs, collect(0:ΔE:E_max))
    # Plot η(E)
    plt1 = plot((0. + ΔE / 2):ΔE:(E_max - ΔE / 2), ηs,
        label=[L"$\Gamma_a$", L"$\Gamma_s$", L"$\Gamma_b$"],
        ylabel=L"$\eta_i$", xlabel=L"$E(A)$", framestyle=:box, ylims=(0., 1.));
    # Plot avg_η(E)
    plt2 = plot((0. + ΔE / 2):ΔE:(E_max - ΔE / 2), η(Γ_regs),
        ylabel=L"$\eta$", xlabel=L"$E(A)$", framestyle=:box, legend=false,
        ylims=(0., 1.));
    # Plot N(E)
    plt3 = plot(ΔE:ΔE:E_max, [length.(Γ_regs[i]) for i in 1:length(Γs)],
        ylabel=L"$N$", xlabel=L"E(A)", framestyle=:box, seriestype=:step,
        label=[L"$\Gamma_a$", L"$\Gamma_s$", L"$\Gamma_b$"], legend=:topleft);
    # Plot ρ = dN/dE (E)
    plt4 = plot(ΔE:ΔE:E_max, [length.(dΓ[i]) ./ ΔE for i in 1:length(Γs)],
        ylabel=L"$\rho=\frac{dN}{dE}$", xlabel=L"E(A)", framestyle=:box, seriestype=:step,
        label=[L"$\Gamma_a$", L"$\Gamma_s$", L"$\Gamma_b$"], legend=:topleft);

    savefig(plt1, "$prefix/eta(E).pdf");
    savefig(plt2, "$prefix/avg_eta(E).pdf");
    savefig(plt3, "$prefix/N(E).pdf");
    savefig(plt4, "$prefix/rho(E).pdf");
end

function main()
    B, D, N, δ, ϵ, ΔE = input_param()

    for δᵢ in δ
        for b in B
            prefix = "../Output/B$b D$D N$N/delta_st_$δᵢ epsilon_$ϵ"
            main(prefix, ΔE)
        end
    end
end

main()
