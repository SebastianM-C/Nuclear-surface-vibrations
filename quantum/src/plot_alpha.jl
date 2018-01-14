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
        γ₁s = Vector{typeof(get_values(prefix, slices, 6))}()
        κs = Vector{typeof(get_values(prefix, slices, 7))}()

        for b in B
            prefix = "../output/B$b D$D N$N/delta_st_$δᵢ epsilon_$ϵ"
            push!(αs, get_values(prefix, slices, 1))
            push!(ηs, get_values(prefix, slices, 3))
            push!(ε_αs, get_values(prefix, slices, 4))
            push!(ε_ηs, get_values(prefix, slices, 5))
            push!(γ₁s, get_values(prefix, slices, 6))
            push!(κs, get_values(prefix, slices, 7))
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
        plt4 = plot(xlabel=L"$B$", ylabel=L"$\gamma_1$", framestyle=:box,
            xlims=(0.2, 0.8))#, ylims=(-0.4, 0))
        plt5 = plot(xlabel=L"$B$", ylabel=L"$\kappa$", framestyle=:box,
            xlims=(0.2, 0.8))
        # Labels for α(B), η(B) and α(η)
        labels = n_slices > 1 ? "slice no.~".*["$i" for i ∈ 1:n_slices] : ""
        # Plot α(B) and η(B)
        scatter!(plt1, B, αs, label=labels, yerr=ε_αs)
        scatter!(plt2, B, ηs, label=labels, yerr=ε_ηs)
        # Plot α(η)
        scatter!(plt3, ηs, αs, label=labels)

        # Labels for γ₁(B) and κ(B)
        labels = repeat([L"$\Gamma_a$", L"$\Gamma_s$", L"$\Gamma_b$"], outer=n_slices).*
            repeat([" slice no.~$i" for i ∈ 1:n_slices], inner=3)
        # Marker shapes for γ₁(B) and κ(B)
        available_shapes = [:circle :rect :diamond :utriangle :dtriangle :cross :xcross :star5 :pentagon]
        shapes = repeat(available_shapes[:,1:n_slices], inner=(1, 3))
        # Marker colors for γ₁(B) and κ(B)
        colors = [:blue :orange :green]
        γ₁ = hcat((vcat(collect.(γ₁s[i])...) for i in 1:length(γ₁s))...)
        scatter!(plt4, B, γ₁', label=labels, shape=shapes, color=colors)
        plot!(plt4, x->0.631111, label="")#"Wigner")
        plot!(plt4, x->2, label="")#"Poisson")
        κ = hcat((vcat(collect.(κs[i])...) for i in 1:length(κs))...)
        scatter!(plt5, B, κ', label=labels, shape=shapes, color=colors)
        plot!(plt5, x->0.245089, label="")#"Wigner")
        plot!(plt5, x->6, label="")#"Poisson")

        prefix ="../output/D$D N$N/delta_st_$δᵢ epsilon_$ϵ"
        if !isdir(prefix)
            mkpath(prefix)
        end
        savefig(plt1, "$prefix/alpha_$slices.pdf")
        savefig(plt2, "$prefix/eta_$slices.pdf")
        savefig(plt3, "$prefix/alpha_eta_$slices.pdf")
        savefig(plt4, "$prefix/skewness_$slices.pdf")
        savefig(plt5, "$prefix/kurtosis_$slices.pdf")
    end
end

main()
