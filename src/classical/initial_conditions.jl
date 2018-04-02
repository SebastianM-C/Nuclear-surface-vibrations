#!/usr/bin/env julia
module InitialConditions
using NLsolve
using NLopt
using JLD
using Plots
using Hamiltonian

export generateInitialConditions

function objective(q::Vector, grad::Vector)
    sqrt.(sum(q.^2))
end

function generateInscribed(E, n)

    function constraint(q::Vector, grad::Vector)
        V(q) - E
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

function findQs(E, n)

    function f!(q, F)
        F[1] = V(q) - E
    end

    q0 = zeros(n, 2)
    qs = generateInscribed(E, n)

    for i=1:n
        result = nlsolve(f!, qs[i,:], autodiff=true, method=:newton, ftol=1e-13)
#         !converged(result) && warn("Did not converge for $i")
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

# TODO: Don't forget to set use_cache=true
function generateInitialConditions(E, n, m, use_cache=false)
    T_range = linspace(0, E, m)

    q0 = zeros(n*m, 2)
    p0 = zeros(n*m, 2)

    for i=1:m
        idx = n*(i-1)+1:n*(i-1)+n
        p0[idx, 1] = linspace(0, √(2 * T_range[i]), n)
        p0[idx, 2] = real(pUpperLimit.(T_range[i]+0im, p0[idx, 1]))

        q0[idx, :] = findQs(E - T_range[i], n)
    end

    A, B, D = readdlm("param.dat")
    prefix = "../output/B$B D$D E$E"
    if use_cache
        if !isfile("$prefix/z0.jld")
            info("Initial conditions file not found. Generating new conditions.")
            q0, p0, N = generateInitialConditions(E, n, m, use_cache=false)
        else
            q0, p0 = load("$prefix/z0.jld", "q0", "p0")
            N = size(q0, 1)
        end
    else
        q0, p0, N = filter_NaNs!(q0, p0, n, m)
        plt = plot(1:N, i->H(q0[i,:], p0[i,:]) - E, xlabel="index",
            ylabel="Energy error", lab="")
        # TODO: enable cache
        # savefig(plt, "$prefix/initial_energy_err.pdf")
        # save("$prefix/z0.jld", q0, "q0", p0, "p0")
    end
    info("The maximum error for the initial conditions is "*
        "$(maximum(abs(H(q0[i,:], p0[i,:]) - E) for i = 1:N))")
    return q0, p0, N
end

end  # module InitialConditions
