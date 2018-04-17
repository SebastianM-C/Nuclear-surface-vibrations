#!/usr/bin/env julia
module DataIO
using JLD

include("regions.jl")

export Γ_regs_i, Γ_regs_idx, Γ_key, read_Γ, add, get_values

function Γ_regs_i(Γ_regs::NTuple{3, AbstractArray{<:AbstractArray}}, i::Int)
    Γ_regs[1][i], Γ_regs[2][i], Γ_regs[3][i]
end

function Γ_regs_idx(Γ_regs::NTuple{3, AbstractArray{<:AbstractArray}}, i::Int)
    Γ_regs[1][i].indexes[1], Γ_regs[2][i].indexes[1], Γ_regs[3][i].indexes[1]
end

function Γ_key(Γ_regs::NTuple{3, AbstractArray{<:AbstractArray}}, i::Int)
    hash(Γ_regs_i(Γ_regs, i)), Γ_regs_idx(Γ_regs, i)
end

function read_Γ(prefix)
    readdlm("$prefix/reuna.dat")[:,1], readdlm("$prefix/reuns.dat")[:,1],
            readdlm("$prefix/rebde.dat")[:,1]
end

function add(prefix, Γ_regs, αs, ηs, avg_ηs, ε_αs, ε_ηs, γ1, kurt)
    if !isdir(prefix)
        mkpath(prefix)
    end
    if isfile("$prefix/stats.jld")
        dict = load("$prefix/stats.jld", "dict")
    else
        dict = Dict{typeof(Γ_key(Γ_regs, 1)),
            typeof((αs[1], ηs[1], avg_ηs[1], ε_αs[1], ε_ηs[1], γ1[1], kurt[1]))}()
    end

    for i=1:length(Γ_regs[1])
        dict[Γ_key(Γ_regs, i)] = (αs[i], ηs[i], avg_ηs[i], ε_αs[i], ε_ηs[i],
            γ1[i], kurt[i])
    end

    save("$prefix/stats.jld", "dict", dict)

end

function load_dict_regs(prefix::AbstractString, slices)
    dict = load("$prefix/stats.jld", "dict")
    Γs = read_Γ(prefix)
    Γ_regs = Regions.regions(Γs, slices)
    return dict, Γ_regs
end

function get_values(prefix::AbstractString, slices)
    dict, Γ_regs = load_dict_regs(prefix, slices)
    vals = Vector{typeof(dict[Γ_key(Γ_regs, 1)])}()
    for i=1:length(Γ_regs[1])
        push!(vals, dict[Γ_key(Γ_regs, i)])
    end
    vals
end

function get_values(prefix::AbstractString, slices, idx)
    dict, Γ_regs = load_dict_regs(prefix, slices)
    if slices == 1
        return dict[Γ_key(Γ_regs, 1)][idx]
    end
    vals = Vector{typeof(dict[Γ_key(Γ_regs, 1)][idx])}()
    for i=1:length(Γ_regs[1])
        push!(vals, dict[Γ_key(Γ_regs, i)][idx])
    end
    vals
end

end  # module DataIO
