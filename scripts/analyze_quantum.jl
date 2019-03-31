include("energylevels.jl")
include("../utils.jl")
include("plots.jl")

using EnergyLevels
using Utils

using StatsBase, LaTeXStrings
using Plots, StatPlots
using DataFrames, Query
using JLD

function compare_δ(;n1=500,f1=0.075,n2=520,f2=f1,b=0.55)
    E1, E2 = EnergyLevels.resize(elvls(n2, f2, b=b), elvls(n1, f1, b=b))
    bar(abs.(E1 - E2), lc=nothing, yscale=:log10) |> display
    maximum(abs.(E1 - E2))
end

df = concat(r"perf_data.csv")
data = add_max_δ(df)

pgfplots()
# data |> @filter(_.n==120) |> @map(makeplots(120, _.b))
for r in eachrow(data[(data[:n].==500),:])
     makeplots(500, b=r[:b], slices=2)
end


makeplots(500, b=0.1)

params = concat(r"fit_data.csv")
params |> @orderby(_.b) |>
    @filter(_.n==500 && _.slices == 1 && _.slice_idx==1 && _.b ≥ 0.2) |>
    @map({_.b, _.α, _.avg_η, q=1-_.q, z=1-_.z}) |>
    @df plot(:b, [:α, :avg_η, :q, :z], m=4,
        label=[L"$\alpha$" L"$\overline{\eta}$" L"Brody $1-q$" L"Berry $1-z$"],
        ylabel="Fit Parameters", xlabel=L"$B$", framestyle=:box)

savefig("../../output/quantum/n500-d0.4/fit_comparison_0.2.pdf")

params |> @orderby(_.b) |>
    @filter(_.n==500 && _.slices == 2 && _.slice_idx==1) |>
    @map({_.b, _.α}) |>
    @df plot(:b, :α, m=4, label="slice 1", xlabel=L"B", ylabel=L"$\alpha$")
params |> @orderby(_.b) |>
    @filter(_.n==500 && _.slices == 2 && _.slice_idx==2) |>
    @map({_.b, _.α}) |>
    @df plot!(:b, :α, m=4, label="slice 2")
savefig("../../output/quantum/n500-d0.4/alpha_slices.pdf")

params |> @orderby(_.b) |>
    @filter(_.n==500 && _.slices == 2 && _.slice_idx==1) |>
    @map({_.b, q=1-_.q}) |>
    @df plot(:b, :q, m=4, label="slice 1", xlabel=L"B", ylabel=L"$1-q$")
params |> @orderby(_.b) |>
    @filter(_.n==500 && _.slices == 2 && _.slice_idx==2) |>
    @map({_.b, q=1-_.q}) |>
    @df plot!(:b, :q, m=4, label="slice 2")
savefig("../../output/quantum/n500-d0.4/brody_slices.pdf")



test = DataFrame()
test[:a] = [1,2,3,4]
test[:b] = [1,1,2,3]
test |> @filter(_.b==1) |> @map(println(_.a))
for r in eachrow(test[test[:b].==1 .& test[:a].<2,:])
     println(r[:a])
end

compare_δ(b=0.55,n2=550)

# Problems with: 0.1, 0.12, 0.2, 0.25, 0.27 (2), 0.32, 0.42, 0.5

b = 0.1
n = 500
E, eigv = diagonalize(n, b=b)

bin_size = 0.2
slices = 1
Γs = irreducible_reps(E, eigv, n)
Γ_regs = regions(Γs, slices)
symm, Δ = EnergyLevels.filter_symmetric(eigv, n, ϵ=1e-3)
ΔE, bd = EnergyLevels.filter_bidimensional(E, ε=1e-9)
E[symm .& bd]

count(symm .& bd)
E[(E.>379.8) .& (E.<380.3)]






E[symm .& (E.>379) .& (E.<381)]


Δ[symm .& (E.>380) .& (E.<380.3)]

E[ΔE .& (E.>380) .& (E.<380.3)]


push!(diff(E), 10)[ΔE .& (E.>380) .& (E.<380.3)]



histogram(Δ[(Δ .<= 1e-3) .& .!bd], bins=logspace(-14, 1, 16), xscale=:log10,
    xlims=(1e-14, 10), xlabel=L"R_y |\Psi\rangle - |\Psi\rangle",
    ylabel=L"N")
histogram!(Δ[(Δ .> 1e-3) .& .!bd], bins=logspace(-14, 1, 16))
histogram!(Δ[bd], bins=logspace(-14, 1, 16), α=0.4, ylims=(0,600))

E[(Δ .> 1e-3) .& (Δ .< 1e-2) .& .!bd]
push!(diff(E), 10)

########### Compare with old results

include("plot_statistics.jl")

plt, old_plt, e_diffs = compare_old(b, n, slices=1)
plot(e_diffs[3][1] - e_diffs[3][2])

function compare_old(b, n; δ=1e-9, ϵ=1e-8, slices=1, bin_size=0.2)
    old_prefix = "../../output/quantum/B$b D0.4 N$n/delta_st_$δ epsilon_$ϵ"
    old_Γ = DataIO.read_Γ(old_prefix)[end:-1:1]

    E, eigv = diagonalize(n, b=b)
    Γs = irreducible_reps(E, eigv, n)
    old_regs = regions(old_Γ, slices)

    plt = plot_P_fit(Γ_regs_i(Γ_regs, 1), bin_size)
    old_plt = plot_P_fit(Γ_regs_i(old_regs, 1), bin_size)
    plt, old_plt, EnergyLevels.resize.(old_Γ, Γs)
end
