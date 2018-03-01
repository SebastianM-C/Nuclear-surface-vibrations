using Plots
using SpecialFunctions
using LaTeXStrings

pgfplots()

R(z) = 1 - exp(π*z^2/4) * erfc(√(π)*z/2)
P_B(q,s) = exp(-(1-q)s - π/4 * q^2 * s^2) * (1-q^2+π/2 * q^3 * s - (1-q)^2 * R(q*s))

P(α,s) = α * exp(-s) + (1 - α) * π / 2 * s * exp(-π / 4 * s^2)

plt=plot(xlabel=L"s", ylabel=L"P", framestyle=:box)
for i=0.15:0.2:1
    plot!(plt, 0:0.02:4, x->P_B(i,x), ls=:dash, lw=3)
    plot!(plt, 0:0.02:4, x->P(1-i,x), lw=3)
end
plt

savefig(plt, "comparison.tex")

pwd()
