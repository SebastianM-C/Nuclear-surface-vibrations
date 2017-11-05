#!/usr/bin/env julia
module Regions

export regions, cumulative_regions, fix_regs

function regions(Γ::AbstractArray, endpoints::AbstractVector{Int})
    n = length(endpoints)
    @assert n ≥ 2 "There must be at least 2 endpoints."
    [@view Γ[endpoints[i]+1:endpoints[i+1]] for i in 1:n-1]
end

function regions(Γ::AbstractArray, n::Integer)
    regions(Γ, count_to_index(Γ, n))
end

function regions(Γ::AbstractArray{T}, E_list::Vector{T}) where {T}
    regions(Γ, energy_to_index(Γ, E_list))
end

function regions(Γs::NTuple{N, T}, slices) where {N, T}
    ntuple(i->regions(Γs[i], slices), length(Γs))
end

function cumulative_regions(Γ::AbstractArray{T}, ΔE::T) where {T}
    E_max = ceil(Γ[end] / ΔE) * ΔE
    E_slices = ΔE:ΔE:E_max
    [regions(Γ, [zero(eltype(Γ)), Eᵢ])[1] for Eᵢ in E_slices]
end

function cumulative_regions(Γs::NTuple{N, T}, ΔE::Real) where {N, T}
    ntuple(i->cumulative_regions(Γs[i], ΔE), length(Γs))
end

function fix_regs(Γ_regs)
    maxlen = maximum(length.(Γ_regs))
    minlen = minimum(length.(Γ_regs))
    if maxlen ≠ minlen
        for i in 1:length(Γ_regs)
            if length(Γ_regs[i]) > minlen
                deleteat!(Γ_regs[i], length(Γ_regs[i])-1)
            end
        end
    end
end

function count_to_index(Γ::AbstractArray, n::Integer)
    Δ = Int(floor(size(Γ, 1) / n))
    endpoints = [Δ * i for i in 0:n]
    endpoints[end] = length(Γ)
    return endpoints
end

function energy_to_index(Γ::AbstractArray{T}, endpoints::Vector{T}) where {T}
    n = size(endpoints, 1)
    @assert n ≥ 2 "There must be at least 2 endpoints."
    idx = [indmin(abs.(Γ .- i)) - 1 for i in endpoints]
    idx[end] = indmin(abs.(Γ .- endpoints[end]))
    return idx
end

end  # module Regions
