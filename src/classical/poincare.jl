module Poincare

export poincaremap, coloredpoincare

using Distributed
using ..InitialConditions

using OrdinaryDiffEq
using DynamicalSystemsBase
using ChaosTools
using StaticArrays
using Plots, LaTeXStrings

using ..Hamiltonian

"""
    poincaremap(q0, p0; params=(A=1, B=0.55, D=0.4), t=500., axis=3, sgn=1,
            diff_eq_kwargs=(abstol=1e-14,reltol=0,maxiters=1e9), full=false)

Create a Poincaré map at the given energy for the given parameters through
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
function poincaremap(q0, p0; params=PhysicalParameters(), t=500., axis=3, sgn=1,
        diff_eq_kwargs=(alg=Vern9(), abstol=1e-14, reltol=0, maxiters=1e9),
        rootkw=(xrtol=1e-6, atol=1e-6), full=false)
    q0[:,1] .+= eps()
    z0 = [SVector{4}(vcat(p0[i, :], q0[i, :])) for i ∈ axes(q0, 1)]
    idxs = full ? SVector{4}(1:4) : (axis==3) ? SVector{2}([4,2]) : SVector{2}([3,1])

    ds = ContinuousDynamicalSystem(Hamiltonian.ż, z0[1], params)
    integ = DynamicalSystemsBase.integrator(ds; diff_eq_kwargs...)
    output = pmap(eachindex(z0)) do i
        DynamicalSystemsBase.reinit!(integ, z0[i])
        ChaosTools.poincaresos(integ, ChaosTools.PlaneCrossing((axis, 0), false), t, 0, idxs, rootkw)
    end

    return output
end

function coloredpoincare(E, colors;
        name="", alg=PoincareRand(n=50), params=PhysicalParameters(),
        t=500., sgn=1, axis=3,
        diff_eq_kwargs=(abstol=1e-14,reltol=0,maxiters=1e9))
    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    axis = !isa(alg, InscribedCircle) ? (isa(alg.plane, InitialConditions.Symmetric) ? 3 : 4) : axis
    q0, p0 = initial_conditions(E, alg=alg, params=params)
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

end  # module Poincare
