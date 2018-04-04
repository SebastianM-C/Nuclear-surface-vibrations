include("energylevels.jl")
# include("hamiltonian.jl")

using EnergyLevels
# using Hamiltonian
using Plots
using TimerOutputs

n = 120
f = 0.1

# N = n*(n+1)/2
# nev = Int(floor(f*N))
# @time E, eigv, nconv, niter, nmult, resid = eigs(H, nev=nev, which=:SM)

levels(n, f)

print_timer()
# maximum(abs.(sort(real(E)) - E2[:values][1:length(E)]))
# eps(10000.)
# E2, eigv2, max_c_idx2, index2 = levels(140, 0.7)
#
# plot(eigv2[1:size(eigv,1),908])
# plot(E2[1:length(E)] - E)
# maximum(abs.(E2[1:length(E)] - E))
