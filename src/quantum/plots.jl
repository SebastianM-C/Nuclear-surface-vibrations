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

    plot_err(E, eigv, n, prefix)

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
            fit_histogram(x, data, model).param[1], Γ_regs_i(Γ_regs, i)[1][1],
            Γ_regs_i(Γ_regs, i)[1][end], Γ_regs_i(Γ_regs, i)[2][1],
            Γ_regs_i(Γ_regs, i)[2][end], Γ_regs_i(Γ_regs, i)[3][1],
            Γ_regs_i(Γ_regs, i)[3][end],
            ηs..., η(Γ_regs_i(Γ_regs, i)),
            γ1..., κ...])
        unique!(df, :region)
    end
    CSV.write("$prefix/../fit_data.csv", df)
end

function plot_hist(Γ_regs, slices, prefix; bin_size=0.2)
    for i=1:length(Γ_regs[1])
        data = rel_spacing.(Γ_regs_i(Γ_regs, i))
        saveparams(prefix, 0:bin_size:4, data, Γ_regs, slices, i)
        plt = fithistogram(0:bin_size:4, data, model,
            xlabel=L"$s$", ylabel=L"$P(s)$")
        fn = "$prefix/P(s)_slice_$i-of-$slices.pdf"
        savefig(plt, fn)
    end
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
