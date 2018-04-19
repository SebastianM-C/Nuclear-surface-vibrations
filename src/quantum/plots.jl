include("regions.jl")
include("statistics.jl")
include("recipes.jl")

using Plots, LaTeXStrings
using Regions, Statistics
using Recipes

function makeplots(n, b=0.55, d=0.4; ϵ=1e-6, ε=1e-9, slices=1, bin_size=0.2)
    E, eigv = diagonalize(n, b=b, d=d)
    Γs = irreducible_reps(E, eigv, n, ϵ=ϵ, ε=ε)

    prefix = "../../output/quantum/n$n-b$b-d$d/"

    plot_err(E, eigv, n, prefix);

    Γ_regs = regions(Γs, slices)
    prefix *= "eps$ϵ-veps$ε"

    # HACK: Workaround for https://github.com/JuliaStats/StatsBase.jl/issues/315
    fail_count = 0
    try
        plot_hist(Γ_regs, prefix, bin_size=bin_size)
    catch InexactError()
        fail_count += 1
        fail_count < 100 && plot_hist(Γ_regs, prefix, bin_size=bin_size)
    end

    fail_count == 0 && info("Failed $fail_count times.")

    return nothing
end

function plot_hist(Γ_regs, prefix; bin_size=0.2)
    for i=1:length(Γ_regs[1])
        plt = fithistogram(0:bin_size:4, rel_spacing.(Γ_regs_i(Γ_regs, 1)), model,
            xlabel=L"$s$", ylabel=L"$P(s)$")
        fn = replace("$(Γ_regs_idx(Γ_regs, 1))", r":", s"-")
        replace(fn, r", ", s"_")
        fn = "$prefix/P(s)_$fn.pdf"
        savefig(plt, fn)
    end
    plt
end

function plot_err(E, eigv, n, prefix)
    symm, Δ = EnergyLevels.filter_symmetric(E, eigv, n)
    s = diff(E)

    plt1 = histogram(s, bins=logspace(-14, 1, 16), xscale=:log10,
        xlims=(1e-14, 10), xlabel=L"E_{n+1} - E_n", ylabel=L"N")
    plt2 = histogram(Δ, bins=logspace(-14, 1, 16), xscale=:log10,
        xlims=(1e-14, 10), xlabel=L"R_y |\Psi\rangle - |\Psi\rangle",
        ylabel=L"N")
    savefig(plt1, "$prefix/bd_err.pdf")
    savefig(plt2, "$prefix/symm_err.pdf")

    plt1, plt2
end
