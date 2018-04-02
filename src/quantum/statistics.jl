#!/usr/bin/env julia
module Statistics

export rel_spacing, hist_P, fit_α, η, model, staircase

using StatsBase
using Plots, LaTeXStrings
using LsqFit, QuadGK

include("dataio.jl")
using .DataIO

function rel_spacing(Γ::AbstractArray)
    n = length(Γ)
    [n * (Γ[i + 1] - Γ[i]) / (Γ[end] - Γ[1]) for i in 1:n - 1]
end

wigner(s) = π / 2 * s * exp(-π / 4 * s^2)
poisson(s) = exp(-s)
model(s, α) = @. α[1] * poisson(s) + (1 - α[1]) * wigner(s)

function hist_P(Γ, bin_size)
    data = rel_spacing.(Γ)
    ws = @. weights(ones(Γ) / (3 * length(Γ) * bin_size))
    hists = [fit(Histogram, d, w, 0:bin_size:4, closed=:left)
        for (d, w) in zip(data, ws)]
end

function fit_α(Γ::NTuple{N, AbstractArray{<:Number}}, bin_size) where {N}
    fit_α(hist_P(Γ, bin_size), bin_size)
end

function fit_α(Γs::NTuple{N, AbstractArray{<:AbstractArray}}, bin_size) where {N}
    [fit_α(Γ_regs_i(Γs, i), bin_size) for i in 1:length(Γs[1])]
end

function fit_α(hists::AbstractVector, bin_size)
    nbins = Int(4 / bin_size)
    x_data = linspace(0 + bin_size / 2, 4 - bin_size / 2, nbins)
    y_data = sum(h.weights for h in hists)
    p0 = rand(1,)
    c_fit = curve_fit(model, x_data, y_data, ones(nbins), p0;
        lower=[0.], upper=[1.])
    return c_fit
end

function η(Γᵢ)
    σ = var(rel_spacing(Γᵢ))    # the varaince of the relative spacings
    σ_W = 4 / π - 1
    σ_P = 1
    (σ - σ_W) / (σ_P - σ_W)
end

function η(Γ::NTuple{N, AbstractArray{<:Number}}) where {N}
    sum(length.(Γ) .* η.(Γ)) / sum(length.(Γ))
end

function η(Γs::NTuple{N, AbstractArray{<:AbstractArray}}) where {N}
    [η(Γ_regs_i(Γs, i)) for i in 1:length(Γs[1])]
end

end  # module Statistics
