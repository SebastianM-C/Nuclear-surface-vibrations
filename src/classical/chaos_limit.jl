#!/usr/bin/env julia
module ChaosLimit
using StatsBase

export chaos_limit

function chaos_limit(λs)
    n = size(λs, 1)
    h = fit(Histogram, λs, closed=:left, nbins=floor(n/4))
    idx = findin(h.weights, 0)[1]
    ch_lim = collect(h.edges[1])[idx]
    n_chaotic = count(λs .> ch_lim)
    if n_chaotic / n > 0.9 && maximum(λs) < 5e-2
        return maximum(λs), 0
    else
        return ch_lim, n_chaotic
    end
end

end  # module ChaosLimit
