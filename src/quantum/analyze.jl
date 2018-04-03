include("energylevels.jl")

using EnergyLevels
using Plots

nlvl(n,f) = Int(floor(n*(n+1)/2*f))
nlvl(340,0.05)
E, eigv, max_c_idx, index = levels(340, 0.05)

E2, eigv2, max_c_idx2, index2 = levels(140, 0.7)

plot(eigv2[1:size(eigv,1),908])
plot(E2[1:length(E)] - E)
maximum(abs.(E2[1:length(E)] - E))
