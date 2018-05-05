module CustomPlots

export energy_error

using RecipesBase
using DiffEqBase
using Plots

include("hamiltonian.jl")
using .Hamiltonian

function energy_error(sim, E, params)
    energy_err(t,u1,u2,u3,u4) = (t, H([u1,u2],[u3,u4], params) - E)
    energy_err(sol) = size(sol.u,1) > 0 ?
        [abs.(H([sol[1,j], sol[2,j]], [sol[3,j], sol[4,j]], params) - E)
        for j=1:size(sol,2)] : 0
    info("The maximum energy error during time evolution was "*
        "$(maximum(map(i->maximum(energy_err(i)), sim.u)))")

    plt = plot(ylabel="Energy error", legend=false)
    plot!(plt, sim, vars=(energy_err, 0,1,2,3,4), msc=nothing, ms=2)
end

@recipe function f(sim::AbstractMonteCarloSolution;
                   zcolors = typeof(sim.u)<:AbstractArray ? fill(nothing, length(sim.u)) : nothing,
                   idxs = typeof(sim.u)<:AbstractArray ? eachindex(sim.u) : 1)
  for i in idxs
    size(sim[i].u, 1) == 0 && continue
    @series begin
      legend := false
      xlims --> (-Inf,Inf)
      ylims --> (-Inf,Inf)
      zlims --> (-Inf,Inf)
      zcolor --> zcolors[i]
      sim[i]
    end
  end
end

end  # module CustomPlots
