# module Poincare

# export poincaremap, coloredpoincare

# using ..InitialConditions
# using ..Distributed

# @everywhere include("$(@__DIR__)/hamiltonian.jl")
using ChaosTools
using StaticArrays
using Plots, LaTeXStrings
# @everywhere using .Hamiltonian

"""
    poincaremap(E; n=10, m=10, A=1, B=0.55, D=0.4, t=100, axis=3)

Create a Poincare map at the given energy for the given parameters through
a Monte Carlo simulation.

## Arguments
- `E`: The energy of the system
- `n = 10`: Target number of initial conditions per kinetic energy values
- `m = 10`: Target number of kinetic energy values
- `A = 1`: Hamiltonian A parameter
- `B = 1`: Hamiltonian B parameter
- `D = 1`: Hamiltonian D parameter
- `t = 100`: Simulation duration
- `axis = 3`: Axis for the Poincare section
- `sgn = 1`: The intersection direction with the plane
"""
function poincaremap(q0, p0; params=(A=1, B=0.55, D=0.4), t=500., axis=3, sgn=1,
        diff_eq_kwargs=(abstol=1e-14,reltol=0,maxiters=1e9), full=false)
    # temp fix for poincaresos special case
    q0[:,1] .+= eps()
    z0 = [StaticArrays.SVector{4}(vcat(p0[i, :], q0[i, :])) for i ∈ axes(q0, 1)]
    idxs = full ? (1:4) : (axis==3) ? [4,2] : [3,1]
    output = pmap(eachindex(z0)) do i
        @debug "idx" i
        ds = ChaosTools.ContinuousDynamicalSystem(Hamiltonian.ż, z0[i], params)
        ChaosTools.poincaresos(ds, (axis, 0), t; direction=sgn, idxs=idxs,
            diff_eq_kwargs...)
    end

    return output
end

function coloredpoincare(E, colors;
        name="", n=500, m=nothing, alg=Val(:poincare_rand), symmetric=Val(true),
        border_n=1000, recompute=false, params=(A=1, B=0.55, D=0.4),
        t=500., sgn=1,
        diff_eq_kwargs=(abstol=1e-14,reltol=0,maxiters=1e9))
    prefix = "output/classical/B$B-D$D/E$E"
    axis = isa(symmetric, Val{true}) ? 3 : 4
    q0, p0 = initial_conditions(E, n, m, params=(A,B,D), alg=alg,
        symmetric=symmetric, border_n=border_n, recompute=recompute)
    sim = poincaremap(q0, p0; params=params, t=t, axis=axis, sgn=sgn,
        diff_eq_kwargs=diff_eq_kwargs)

    plt = plot()
    if axis == 3
        xlabel =  L"$q_2$"
        ylabel =  L"$p_2$"
    else
        xlabel = L"$q_1$"
        ylabel = L"$p_1$"
    end
    for (i,s) in enumerate(sim)
        plt = scatter!(plt, s[:,1], s[:,2], ms=1.2, msa=0, label="",
            xlabel=xlabel, ylabel=ylabel, zcolor=colors[i])
    end
    savefig(plt,  "$prefix/poincare_$name-ax$axis-t_$t-_sgn$sgn.pdf")
end

# end  # module Poincare
