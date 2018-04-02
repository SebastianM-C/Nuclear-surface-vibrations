#!/usr/bin/env julia
using Plots, LaTeXStrings
pgfplots()

include("input.jl")
include("dataio.jl")
using DataIO

function main()
    B, D, N, δ, ϵ, bin_size, slices = input_param()

    for δᵢ in δ
        prefix = "../output/B$(B[1]) D$D N$N/delta_st_$δᵢ epsilon_$ϵ"
        αs = Vector{typeof(get_values(prefix, slices, 1))}()
        ηs = Vector{typeof(get_values(prefix, slices, 3))}()
        ε_αs = Vector{typeof(get_values(prefix, slices, 4))}()
        ε_ηs = Vector{typeof(get_values(prefix, slices, 5))}()

        for b in B
            prefix = "../output/B$b D$D N$N/delta_st_$δᵢ epsilon_$ϵ"
            push!(αs, get_values(prefix, slices, 1))
            push!(ηs, get_values(prefix, slices, 3))
            push!(ε_αs, get_values(prefix, slices, 4))
            push!(ε_ηs, get_values(prefix, slices, 5))
        end

        n_slices = length(αs[1])
        αs = hcat(αs...)'
        ηs = hcat(ηs...)'
        ε_αs = hcat(ε_αs...)'
        ε_ηs = hcat(ε_ηs...)'

        # Plots setup
        plt1 = plot(xlabel=L"$B$", ylabel=L"$\alpha$", framestyle=:box,
            xlims=(0.2, 0.8), ylims=(0., 1.))
        plt2 = plot(xlabel=L"$B$", ylabel=L"$\eta$", framestyle=:box,
            xlims=(0.2, 0.8), ylims=(0., 1.))
        plt3 = plot(xlabel=L"$\eta$", ylabel=L"$\alpha$", framestyle=:box,
            xlims=(0., 1.), ylims=(0., 1.), aspect_ratio=1)
        # Labels for α(B), η(B) and α(η)
        labels = n_slices > 1 ? "slice no.~".*["$i" for i ∈ 1:n_slices] : ""
        # Plot α(B) and η(B)
        scatter!(plt1, B, αs, label=labels, yerr=ε_αs)
        scatter!(plt2, B, ηs, label=labels, yerr=ε_ηs)
        # Plot α(η)
        scatter!(plt3, ηs, αs, label=labels)

        # Save the plots
        prefix ="../output/D$D N$N/delta_st_$δᵢ epsilon_$ϵ"
        if !isdir(prefix)
            mkpath(prefix)
        end
        savefig(plt1, "$prefix/alpha_$slices.pdf")
        savefig(plt2, "$prefix/eta_$slices.pdf")
        savefig(plt3, "$prefix/alpha_eta_$slices.pdf")
    end
end

main()
