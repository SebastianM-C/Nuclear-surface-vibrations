include("regions.jl")
include("statistics.jl")
include("recipes.jl")
include("dataio.jl")


using Plots, LaTeXStrings
using Regions, DataIO, Statistics
using Recipes

function makeplots(n, b=0.55, d=0.4; ϵ=1e-6, ε=1e-9, slices=1, bin_size=0.2)
    E, eigv = diagonalize(n, b=b, d=d)
    Γs = irreducible_reps(E, eigv, n, ϵ=ϵ, ε=ε)

    prefix = "../../output/quantum/n$n-b$b-d$d/"

    plot_err(E, eigv, n, prefix, ϵ=ϵ, ε=ε)

    Γ_regs = regions(Γs, slices)
    prefix *= "eps$ϵ-veps$ε"
    if !isdir(prefix)
        mkpath(prefix)
    end

    # HACK: Workaround for https://github.com/JuliaStats/StatsBase.jl/issues/315
    fail_count = 0
    try
        plot_hist(Γ_regs, slices, prefix, bin_size=bin_size)
    catch InexactError()
        fail_count += 1
        fail_count < 100 && plot_hist(Γ_regs, slices, prefix, bin_size=bin_size)
    end

    fail_count != 0 && info("Failed $fail_count times.")

    return nothing
end

function plot_hist(Γ_regs, slices, prefix; bin_size=0.2)
    for i=1:length(Γ_regs[1])
        data = rel_spacing.(Γ_regs_i(Γ_regs, i))
        saveparams(prefix, 0:bin_size:4, data, Γ_regs, slices, i)
        plt = fithistogram(0:bin_size:4, data, model, ([0.],[1.]), "\\alpha",
            label=[L"$\Gamma_b$" L"$\Gamma_s$" L"$\Gamma_a$"],
            xlabel=L"$s$", ylabel=L"$P(s)$")
        cmp = fithistogram(0:bin_size:4, data, [model, brody, berry, lwd],
            [([0.],[1.]), ([0.],[1.]), ([0.],[1.]), ([0.], [Inf])],
            ["\\alpha", "q", "z", "w"],
            label=[L"$\Gamma_b$" L"$\Gamma_s$" L"$\Gamma_a$"],
            xlabel=L"$s$", ylabel=L"$P(s)$")
        fn1 = "$prefix/P(s)_slice_$i-of-$slices.pdf"
        fn2 = "$prefix/P(s)_cmp_slice_$i-of-$slices.pdf"
        savefig(plt, fn1)
        savefig(cmp, fn2)
    end
end

function plot_err(E, eigv, n, prefix; ϵ=1e-6, ε=1e-9)
    symm, Δ = EnergyLevels.filter_symmetric(eigv, n, ϵ=ϵ)
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

function parse_prefix(prefix)
    re = r"n([0-9]+)-b([0-9]+\.?[0-9]+)-d([0-9]+\.?[0-9]+)/eps([0-9]+\.?[0-9]*(?:e-?[0-9]+)?)-veps([0-9]+\.?[0-9]*(?:e-?[0-9]+)?)"
    n, b, d, ϵ, ε = match(re, prefix).captures
    parse(Int, n), float.((b, d, ϵ, ε))...
end

function saveparams(prefix, x, data, Γ_regs, slices, i)
    n, b, d, ϵ, ε = parse_prefix(prefix)
    ηs = η.(Γ_regs_i(Γ_regs, i))
    γ1 = skewness.(data)
    κ = kurtosis.(data)
    if !isfile("$prefix/../fit_data.csv")
        df = DataFrame()
        df[:n] = [n]
        df[:b] = [b]
        df[:d] = [d]
        df[:ϵ] = [ϵ]
        df[:ε] = [ε]
        df[:region] = ["$slices#$i"]
        df[:slices] = [slices]
        df[:slice_idx] = [i]
        df[:E0_Γ₂] = [Γ_regs_i(Γ_regs, i)[1][1]]
        df[:E_Γ₂] = [Γ_regs_i(Γ_regs, i)[1][end]]
        df[:E0_Γₛ] = [Γ_regs_i(Γ_regs, i)[2][1]]
        df[:E_Γₛ] = [Γ_regs_i(Γ_regs, i)[2][end]]
        df[:E0_Γₐ] = [Γ_regs_i(Γ_regs, i)[3][1]]
        df[:E_Γₐ] = [Γ_regs_i(Γ_regs, i)[3][end]]
        df[:α] = [fit_histogram(x, data, model).param[1]]
        df[:q] = [fit_histogram(x, data, brody).param[1]]
        df[:z] = [fit_histogram(x, data, berry).param[1]]
        df[:w] = [fit_histogram(x, data, lwd, ([0.],[Inf])).param[1]]
        df[:η₂] = [ηs[1]]
        df[:ηₛ] = [ηs[2]]
        df[:ηₐ] = [ηs[3]]
        df[:avg_η] = [η(Γ_regs_i(Γ_regs, i))]
        df[:γ1₂] = [γ1[1]]
        df[:γ1ₛ] = [γ1[2]]
        df[:γ1ₐ] = [γ1[3]]
        df[:κ₂] = [κ[1]]
        df[:κₛ] = [κ[2]]
        df[:κₐ] = [κ[3]]
    else
        df = CSV.read("$prefix/../fit_data.csv")
        push!(df, [n, b, d, ϵ, ε,
            "$slices#$i", slices, i,
            Γ_regs_i(Γ_regs, i)[1][1], Γ_regs_i(Γ_regs, i)[1][end],
            Γ_regs_i(Γ_regs, i)[2][1], Γ_regs_i(Γ_regs, i)[2][end],
            Γ_regs_i(Γ_regs, i)[3][1], Γ_regs_i(Γ_regs, i)[3][end],
            fit_histogram(x, data, model).param[1],
            fit_histogram(x, data, brody).param[1],
            fit_histogram(x, data, berry).param[1],
            fit_histogram(x, data, lwd, ([0.],[Inf])).param[1],
            ηs..., η(Γ_regs_i(Γ_regs, i)),
            γ1..., κ...])
        unique!(df, :region)
    end
    CSV.write("$prefix/../fit_data.csv", df)
end
