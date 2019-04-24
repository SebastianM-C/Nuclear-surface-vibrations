#!/usr/bin/env julia
using JLD
using Plots, LaTeXStrings

include("chaos_limit.jl")
using ChaosLimit
pgfplots()
# pyplot()

prefix = "../output/"
A, B, D = readdlm("param.dat")
folders = filter(f->contains(f,"B$B D$D"), readdir(prefix))
files = readdir.(joinpath.(prefix, folders))
n = size(folders, 1)
# Select only the folders containing "lyapunov.jld"
folders = folders[[any(contains.(files[i], "lyapunov.jld")) for i in 1:n]]

tmax = 5e4

α = Array{Float64}(0)
λ = Array{Float64}(0)
λ_err = Array{Float64}(0)
λ_minmax = Array{Tuple{Float64,Float64}}(0)
E_list = Array{Float64}(0)

for f in folders
    if load(joinpath(prefix, f, "lyapunov.jld"), "tmax") == tmax
        E = float(match(r"E(?<E>[0-9]+)", f)[1])
        λs, n = load(joinpath(prefix, f, "lyapunov.jld"), "λs", "n")
        ch_lim, n_chaotic = chaos_limit(λs)
        if n_chaotic == 0
            push!(α, 0)
            push!(λ, median(λs))
            push!(λ_err, std(λs))
            push!(λ_minmax, (median(λs) - minimum(λs), maximum(λs) - median(λs)))
            push!(E_list, E)
            continue
        end
        chaotic = λs[λs .> ch_lim]
        push!(α, n_chaotic / n)
        push!(λ, median(chaotic))
        push!(λ_err, std(chaotic))
        push!(λ_minmax, (median(chaotic) - minimum(chaotic), maximum(chaotic) - median(chaotic)))
        push!(E_list, E)
    end
end

if !isdir(joinpath(prefix, "B$B D$D"))
    mkpath(joinpath(prefix, "B$B D$D"))
end

plt = scatter(E_list, α, xlabel=L"$E(A)$", ylabel=L"$R$", legend=nothing,
    framestyle=:box, ylims=(0., 1.));
# savefig(plt, "$prefix/B$B D$D/ratio-t$tmax.pdf");
savefig(plt, "$prefix/B$B D$D/ratio-t$tmax.tex");

plt = scatter(E_list, λ, yerr=λ_err, xlabel=L"$E(A)$", ylabel=L"$\lambda$",
    legend=nothing, framestyle=:box, ylims=(0., 0.4));
# savefig(plt, "$prefix/B$B D$D/median_lyapunov_std-t$tmax.pdf");
savefig(plt, "$prefix/B$B D$D/median_lyapunov_std-t$tmax.tex");

plt = scatter(E_list, λ, yerr=λ_minmax, xlabel=L"$E(A)$", ylabel=L"median $\lambda$",
    legend=nothing, framestyle=:box, ylims=(0., 0.4));
# savefig(plt, "$prefix/B$B D$D/median_lyapunov-t$tmax.pdf");
savefig(plt, "$prefix/B$B D$D/median_lyapunov-t$tmax.tex");
