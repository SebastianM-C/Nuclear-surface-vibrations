#!/usr/bin/env julia
include("input.jl")
include("regions.jl")
include("dataio.jl")
include("statistics.jl")

using Regions, DataIO, Statistics
using JLD

function getE(Γ_regs, rep, idx::Int)
    [Γ_regs[rep][i][idx] for i=1:length(Γ_regs[rep])]
end

function getE(Γ_regs, rep)
    [Γ_regs[rep][i][end] for i=1:length(Γ_regs[rep])]
end

function main(prefix, bin_size, slices)
    Γs = read_Γ(prefix)
    Γ_regs = regions(Γs, slices)
    # load("$prefix/stats.jld", "dict")
    data = get_values(prefix, slices)

    α = [line[1] for line in data]
    η_a = [line[2][1] for line in data]
    η_s = [line[2][2] for line in data]
    η_b = [line[2][3] for line in data]
    η_avg = [line[3] for line in data]
    γ1_a = [line[4][1] for line in data]
    γ1_s = [line[4][2] for line in data]
    γ1_b = [line[4][3] for line in data]
    kurt_a = [line[5][1] for line in data]
    kurt_s = [line[5][2] for line in data]
    kurt_b = [line[5][3] for line in data]
    E₁_a = getE(Γ_regs, 1, 1)
    E₁_s = getE(Γ_regs, 2, 1)
    E₁_b = getE(Γ_regs, 3, 1)
    Eₙ_a = getE(Γ_regs, 1)
    Eₙ_s = getE(Γ_regs, 2)
    Eₙ_b = getE(Γ_regs, 3)
    ΔE_a = getE(Γ_regs, 1) .- getE(Γ_regs, 1, 1)
    ΔE_s = getE(Γ_regs, 2) .- getE(Γ_regs, 2, 1)
    ΔE_b = getE(Γ_regs, 3) .- getE(Γ_regs, 3, 1)

    vals = hcat(1:length(Γ_regs[1]), α, η_a, η_s, η_b, η_avg,
        γ1_a, γ1_s, γ1_b, kurt_a, kurt_s, kurt_b,
        E₁_a, E₁_s, E₁_b, Eₙ_a, Eₙ_s, Eₙ_b, ΔE_a, ΔE_s, ΔE_b,
        [Γ_regs_idx(Γ_regs, i) for i=1:length(Γ_regs[1])])

    open("$prefix/data_slices$slices.txt", "w") do f
        println(f, "│ # │ α       │ η_a     │ η_s    │ η_b     "*
        "│ η_avg    │ γ1_a     │ γ1_s    │ γ1_b     "*
        "│ kurt_a     │ kurt_s    │ kurt_b     "*
        "| E₁_a    │ E₁_s      │ E₁_b      │ Eₙ_a    "*
        "│ Eₙ_s   │ Eₙ_b   │ ΔE_a    │ ΔE_s    │ ΔE_b    "*
        "│            idx            │")
        show(IOContext(f, limit=false), "text/plain", vals)
    end
    # HACK for ommiting type info
    content = readlines("$prefix/data_slices$slices.txt")
    deleteat!(content, 2)
    open("$prefix/data_slices$slices.txt", "w") do f
        for line in content
            println(f, line)
        end
    end
end

function main()
    B, D, N, δ, ϵ, bin_size, slices = input_param()

    for δᵢ in δ
        for b in B
            prefix = "../Output/B$b D$D N$N/delta_st_$δᵢ epsilon_$ϵ"
            main(prefix, bin_size, slices)
        end
    end
end

main()
