#!/usr/bin/env julia
module InitialConditions

export generateInitialConditions

using Logging
using NLsolve
using NLopt
using Plots
using DataFrames, CSV

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
    @debug "Inscribed circle" r q_min ret_code

    θ = range(0, stop=2π, length=n)

    x = r.*cos.(θ)
    y = r.*sin.(θ)

    hcat(x, y)
end

function findQs(E, n, params)

    function f!(F, q)
        F[1] = V(q, params) - E
        F[2] = 0.
    end

    function j!(J, x)
        Hamiltonian.Vjac!(J, x, params)
        J[2,1] = 0
        J[2,2] = 0
    end

    q0 = zeros(n, 2)
    qs = generateInscribed(E, n, params)

    for i=1:n
        result = nlsolve(f!, j!, qs[i,:], method=:trust_region, iterations=50000, ftol=1e-13)
        @debug "root $i" result
        !converged(result) && @warn "Did not converge for $i"
        q0[i,:] = result.zero
    end

    q0
end

function filter_NaNs!(q0, p0, n, m)
    nan_no = count(isnan.(q0))
    nan_no % 2 != 0 && @error "Can't remove NaN lines"
    bad = .!isnan.(q0)
    q0 = reshape(q0[bad], (n*m - Int(nan_no / 2), 2))
    p0 = reshape(p0[bad], (n*m - Int(nan_no / 2), 2))
    N = n*m - Int(nan_no / 2)
    nan_no != 0 && @info "Found $nan_no NaNs and removed $(n*m-N) initial conditions."
    N == 0 && @error "All initial conditions are invalid!"

    return q0, p0, N
end

pUpperLimit(E, p) = √(2 * E - p^2)

function generateInitialConditions(E, n=15, m=15; params=(1, 0.55, 0.4))
    B, D = params[2:3]
    prefix = "../../output/classical/B$B-D$D/E$E"
    if !isdir(prefix)
        mkpath(prefix)
    end
    if !isfile("$prefix/z0.csv")
        @info "No initial conditions file found. Generating new conditions."
        q0, p0, N = _generateInitialConditions(E, n, m, params=params)
    else
        df = CSV.read("$prefix/z0.csv", allowmissing=:none, use_mmap=!is_windows())
        if df[:n][1] != n || df[:m][1] != m
            @info "Incompatible initial conditions. Generating new conditions."
            q0, p0, N = _generateInitialConditions(E, n, m, params=params)
        else
            q0 = hcat(df[:q0₁], df[:q0₂])
            p0 = hcat(df[:p0₁], df[:p0₂])
            N = size(q0, 1)
        end
    end
    return q0, p0, N
end

function _generateInitialConditions(E, n=15, m=15; params=(1, 0.55, 0.4))
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
    df = DataFrame()
    df[:q0₁] = q0[:,1]
    df[:q0₂] = q0[:,2]
    df[:p0₁] = p0[:,1]
    df[:p0₂] = p0[:,2]
    df[:n] = fill(n, N)
    df[:m] = fill(m, N)
    df[:E] = fill(E, N)
    CSV.write("$prefix/z0.csv", df)
    savefig(plt, "$prefix/initial_energy_err.pdf")

    maxerr = maximum(abs(H(p0[i,:], q0[i,:], params) - E) for i = 1:N)
    @info "The maximum error for the initial conditions is $maxerr"
    return q0, p0, N
end

end  # module InitialConditions
