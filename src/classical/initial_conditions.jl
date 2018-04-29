#!/usr/bin/env julia
module InitialConditions

export generateInitialConditions

using NLsolve
using NLopt
using Plots

include("hamiltonian.jl")
using .Hamiltonian

function objective(q::Vector, grad::Vector)
    sqrt.(sum(q.^2))
end

function generateInscribed(E, n, params)

    function constraint(q::Vector, grad::Vector)
        V(q, params) - E
    end

    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [-Inf, -Inf])

    xtol_rel!(opt, 1e-13)

    min_objective!(opt, objective)
    equality_constraint!(opt, constraint, 1e-13)

    r, q_min, ret_code = optimize(opt, [0., 0.])

    θ = linspace(0, 2π, n)

    x = r.*cos.(θ)
    y = r.*sin.(θ)

    hcat(x, y)
end

function findQs(E, n, params)

    function f!(F, q)
        F[1] = V(q, params) - E
        F[2] = 0.
    end

    q0 = zeros(n, 2)
    qs = generateInscribed(E, n, params)

    for i=1:n
        result = nlsolve(f!, qs[i,:], autodiff=:forward, method=:trust_region, ftol=1e-13)
        # !converged(result) && warn("Did not converge for $i")
        q0[i,:] = result.zero
    end

    q0
end

function filter_NaNs!(q0, p0, n, m)
    nan_no = count(isnan.(q0))
    nan_no % 2 != 0 && error("Can't remove NaN lines")
    bad = .!isnan.(q0)
    q0 = reshape(q0[bad], (n*m - Int(nan_no / 2), 2))
    p0 = reshape(p0[bad], (n*m - Int(nan_no / 2), 2))
    N = n*m - Int(nan_no / 2)
    nan_no != 0 && info("Found $nan_no NaNs and removed $(n*m-N) initial conditions.")
    N == 0 && error("All initial conditions are invalid!")

    return q0, p0, N
end

pUpperLimit(E, p) = √(2 * E - p^2)

function generateInitialConditions(E, n=10, m=10; params=(1, 0.55, 0.4))
    T_range = linspace(0, E, m)

    q0 = zeros(n*m, 2)
    p0 = zeros(n*m, 2)

    for i=1:m
        idx = n*(i-1)+1:n*(i-1)+n
        p0[idx, 1] = linspace(0, √(2 * T_range[i]), n)
        p0[idx, 2] = real(pUpperLimit.(T_range[i]+0im, p0[idx, 1]))

        q0[idx, :] = findQs(E - T_range[i], n, params)
    end

    B, D = params[2:3]
    prefix = "../../output/classical/B$B-D$D/E$E"
    if !isdir(prefix)
        mkpath(prefix)
    end
    q0, p0, N = filter_NaNs!(q0, p0, n, m)
    plt = plot(1:N, i->H(p0[i,:], q0[i,:], params) - E, xlabel="index",
        ylabel="Energy error", lab="")
    savefig(plt, "$prefix/initial_energy_err.pdf")

    maxerr = maximum(abs(H(p0[i,:], q0[i,:], params) - E) for i = 1:N)
    info("The maximum error for the initial conditions is $maxerr")
    return q0, p0, N
end

end  # module InitialConditions
