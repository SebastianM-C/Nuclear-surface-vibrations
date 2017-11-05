#!/usr/bin/env julia
module RandomData
include("regions.jl")

using Plots, StatPlots, LaTeXStrings
using StatsBase, Distributions
using LsqFit, NLsolve

using .Regions

export fit_α, η, PoissonWigner, Rayleigh, Exponential, plot_dist

pgfplots()

function plot_P(bars, bin_size, dist::Distribution)
    label = "Random data: "*
        L"$\alpha = "*"$(fit_α(bars, bin_size).param[1]),"*
        L"\,\eta\ = "*"$(η(bars))"*L"$"
    plt = histogram(bins=collect(0:bin_size:4),
        bars, framestyle=:box, normed=true, xlims=(0., 4.), ylims=(0., 1.),
        xlabel=L"$s$", ylabel=L"$P(s)$", label=label);
    if typeof(dist) <: PoissonWigner
        plot!(plt, linspace(0, 4, 100), pdf.(PoissonWigner(dist.α),
            linspace(0, 4, 100)), label="Probability distribution function")
    else
        plot!(plt, dist, label="Probability distribution function")
    end
    return plt
end

function fit_α(s::AbstractVector, bin_size)
    wigner(s) = π / 2 * s * exp(-π / 4 * s^2)
    poisson(s) = exp(-s)
    model(s, α) = @. α[1] * poisson(s) + (1 - α[1]) * wigner(s)

    ws = weights(ones(s) / (float(length(s)) * bin_size))
    hist = fit(Histogram, s, ws, 0:bin_size:4, closed=:left)
    nbins = Int(4 / bin_size)
    x_data = linspace(0 + bin_size / 2, 4 - bin_size / 2, nbins)
    y_data = hist.weights
    p0 = rand(1,)
    c_fit = curve_fit(model, x_data, y_data, ones(nbins), p0;
        lower=[0.], upper=[1.])
    return c_fit
end

function η(s)
    σ = var(s)
    σ_W = 4 / π - 1
    σ_P = 1
    (σ - σ_W) / (σ_P - σ_W)
end


type PoissonWigner{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    α::T

    PoissonWigner{T}(α::T) where {T} =
        (Distributions.@check_args(PoissonWigner, one(α) ≥ α ≥ zero(α)); new{T}(α))
end

PoissonWigner(α::T) where {T<:Real} = PoissonWigner{T}(α)
PoissonWigner(α::Integer) = PoissonWigner(Float64(α))
PoissonWigner() = PoissonWigner(0.4)

function Base.rand(d::PoissonWigner)
    y = rand(Uniform())
    function f!(x, fvec)
        fvec[1] = d.α * (1 - exp(-x[1])) + (1 - d.α) * (1 - exp(-π / 4 * x[1]^2)) - y
    end

    # function g!(x, fjac)
    #     fjac[1, 1] = d.α * exp(-x[1]) + (1 - d.α) * π / 2 * x * exp(-π / 4 * x[1]^2)
    # end

    nlsolve(f!, rand(1,), autodiff=true).zero[1]
end

function Distributions.pdf(d::PoissonWigner{T}, x::Real) where T<:Real
	d.α * exp(-x) + (1 - d.α) * π / 2 * x * exp(- π / 4 * (x^2))
end

function _plot_dist(bin_size, distribution, spacings, fname)
    plt = plot_P(spacings, bin_size, distribution)
    savefig(plt, fname)
end

function plot_dist(prefix, bin_size, slices, N, distribution, name)
    spacings = rand(distribution, N)
    sp_regions = regions(spacings, slices)
    _plot_dist(bin_size, distribution, spacings, "$prefix/"*name*"_$N.pdf")

    for i in 1:slices
        _plot_dist(bin_size, distribution, sp_regions[i],
            "$prefix/"*name*"_$N-slice$i.pdf")
    end
end

end  # module RandomData
