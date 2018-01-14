#!/usr/bin/env julia
module RandomData
include("regions.jl")
include("dataio.jl")

using StatsBase, Distributions
using LsqFit, NLsolve

using .Regions, .DataIO

export fit_α, η, PoissonWigner, Rayleigh, Exponential, α_range

wigner(s) = π / 2 * s * exp(-π / 4 * s^2)
poisson(s) = exp(-s)
model(s, α) = @. α[1] * poisson(s) + (1 - α[1]) * wigner(s)

# function fit_α(s::AbstractVector{<:Number}, bin_size)
#     ws = weights(ones(s) / (float(length(s)) * bin_size))
#     hist = fit(Histogram, s, ws, 0:bin_size:4, closed=:left)
#     nbins = Int(4 / bin_size)
#     x_data = linspace(0 + bin_size / 2, 4 - bin_size / 2, nbins)
#     y_data = hist.weights
#     p0 = rand(1,)
#     c_fit = curve_fit(model, x_data, y_data, ones(nbins), p0;
#         lower=[0.], upper=[1.])
#     return c_fit
# end

function hist_P(data, bin_size)
    ws = @. weights(ones(data) / (3 * length(data) * bin_size))
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

function η(s)
    σ = var(s)
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

function α_range(target_αs, N, bin_size, slices)
    αs = Array{eltype(target_αs)}(length(target_αs), 1)
    ηs = zeros(αs)

    for (idx, target_α) in enumerate(target_αs)
        spacings = rand(PoissonWigner(target_α), N)
        ηs[idx, 1] = η(spacings)
        αs[idx, 1] = fit_α(spacings, bin_size).param[1]

        sp_regions = regions(spacings, slices)
        for i in 1:slices
            ηs[idx, 1+i] = η(sp_regions[i])
            αs[idx, 1+i] = fit_α(sp_regions[i], bin_size).param[1]
        end
    end

    return αs, ηs
end

function α_range(target_αs, N, bin_size)
    αs = Array{eltype(target_αs)}(0)
    ηs = zeros(αs)

    for (idx, target_α) in enumerate(target_αs)
        spacings = rand(PoissonWigner(target_α), N)
        push!(ηs, η(spacings))
        push!(αs, fit_α(spacings, bin_size).param[1])
    end

    return αs, ηs
end

function rand_spacings(Γ_regs::NTuple{N, AbstractArray{<:AbstractArray}}, f, αs) where {N}
    ntuple(i->rand_spacings(Γ_regs[i], f, αs), length(Γ_regs))
end

function rand_spacings(Γ_regs::AbstractArray{<:AbstractArray}, f, αs)
    [rand(PoissonWigner(α), f * length(Γ)) for (α, Γ) in zip(αs, Γ_regs)]
end

end  # module RandomData
