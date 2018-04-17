#!/usr/bin/env julia
using JLD
using Plots, LaTeXStrings

include("chaos_limit.jl")
using ChaosLimit
pgfplots()
# pyplot()
# plotly()

prefix = "../output/"
A, B, D = readdlm("param.dat")
folders = filter(f->contains(f,"B$B D$D"), readdir(prefix))
files = readdir.(joinpath.(prefix, folders))
n = size(folders, 1)
# Select only the folders containing "lyapunov.jld"
folders = folders[[any(contains.(files[i], "lyapunov.jld")) for i in 1:n]]

tmax = 5e4

ch_lim = Array{Float64}(0)
E_list = Array{Float64}(0)

for f in folders
    if load(joinpath(prefix, f, "lyapunov.jld"), "tmax") == tmax
        E = float(match(r"E(?<E>[0-9]+)", f)[1])
        λs = load(joinpath(prefix, f, "lyapunov.jld"), "λs")
        push!(ch_lim, chaos_limit(λs)[1])
        push!(E_list, E)
    end
end

if !isdir(joinpath(prefix, "B$B D$D"))
    mkpath(joinpath(prefix, "B$B D$D"))
end

plt = scatter(E_list, ch_lim, xlabel=L"$E(A)$", ylabel="Chaos limit",
    framestyle=:box);
savefig(plt, "$prefix/B$B D$D/chaos_lim-t$tmax.pdf");
