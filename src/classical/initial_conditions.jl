#!/usr/bin/env julia
module InitialConditions

export generateInitialConditions

using Logging
using Random
using NLsolve
using NLopt
using Roots
using Plots
using DataFrames, CSV
using GeometricalPredicates

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

    θ = range(0+0.1, stop=2π+0.1, length=n)

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
        result = nlsolve(f!, j!, qs[i,:], method=:trust_region, ftol=1e-13)
        !converged(result) && @debug "Did not converge for $i" result
        q0[i,:] = result.zero
    end

    q0
end

function filter_NaNs!(q0, p0, n, m)
    nan_no = count(isnan.(q0))
    if nan_no > 0
        nan_no % 2 != 0 && @error "Can't remove NaN lines" nan_no
        not_NaN = .!isnan.(q0)
        q0 = reshape(q0[not_NaN], (n*m - nan_no ÷ 2, 2))
        p0 = reshape(p0[not_NaN], (n*m - nan_no ÷ 2, 2))
        @info "Found $nan_no NaNs leaving $(n*m-nan_no÷2) initial conditions."
    end
    N = n*m - nan_no ÷ 2
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

function _generateInitialConditions(E, n, m, alg::Val{:inscribed_circle};
        params=(A=1, B=0.55, D=0.4))
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

"""
    tointerval(x, ex)

Resale and move `x` to the interval ``[1, 2)`` given the extrema `ex` of the original interval.
This is required by the algorithms in `GeometricalPredicates`.
"""
function tointerval(x, ex)
    (x - ex[1]) / (ex[2] - ex[1] + 1e-14) + 1
end

function tointerval(points, ex, ey)
    [Point(tointerval(p._x, ex), tointerval(p._y, ey)) for p in points]
end

"""
    frominterval(x, ex)

Move `x` to the original interval with the extrema `ex` from ``[1, 2)``.
See also [`tointerval`](@ref).
"""
function frominterval(x, ex)
    (x - 1) * (ex[2] - ex[1] + 1e-14) + ex[1]
end

function frominterval(points, ex, ey)
    [frominterval(p._x, ex) for p in points], [frominterval(p._y, ey) for p in points]
end

function phase_space_border(E, symmetric::Val{true}, n=1000; param=(A=1, B=0.55, D=0.4))
    p = range(0, stop=√(2 * param.A * E), length=n)
    q0 = E < 1e5 ? 0 : 10
    q = [find_zero(q->H([0,pᵢ], [0,q], param) - E, q0) for pᵢ in p]
    points = Point.(q, p) ∪ Point.(-q[end-1:-1:1], p[end-1:-1:1]) ∪
        Point.(-q[2:end], -p[2:end]) ∪ Point.(q[end-1:-1:2], -p[end-1:-1:2])
    ex = extrema(p._x for p in points)
    ey = extrema(p._y for p in points)
    Polygon(tointerval(points, ex, ey)...), ex, ey
end

function phase_space_border(E, symmetric::Val{false}, n=1000; param=(A=1, B=0.55, D=0.4))
    p = range(0, stop=√(2 * param.A * E), length=n)
    p = p[1:end-1]
    q₊0 = E < 1e5 ? (0,100) : (0,10000)
    q₋0 = E < 1e5 ? (-100,0) : (-10000,0)
    q₊ = [find_zero(q->H([pᵢ,0], [q,0], param) - E, q₊0) for pᵢ in p]
    q₋ = [find_zero(q->H([pᵢ,0], [q,0], param) - E, q₋0) for pᵢ in p]
    points = Point.(q₊, p) ∪ Point.(q₋[end-1:-1:1], p[end-1:-1:1]) ∪
        Point.(q₋[2:end], -p[2:end]) ∪ Point.(q₊[end-1:-1:2], -p[end-1:-1:2])
    ex = extrema(p._x for p in points)
    ey = extrema(p._y for p in points)
    Polygon(tointerval(points, ex, ey)...), ex, ey
end

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Point2D}) = Point((rand(rng) + 1), (rand(rng) + 1))

function _generateInitialConditions(E, n, alg::Val{:poincare_rand},
        symmetric=Val(true); param=(A=1, B=0.55, D=0.4), border_n=1000)
    border, ex, ey = phase_space_border(E, symmetric, border_n, param=param)
    counter = 0
    points = Vector{Point2D}()
    while counter < n
        p = rand(Point2D)
        inpolygon(border, p) && (push!(points, p); counter += 1)
    end
    frominterval(points, ex, ey)
end

function _generateInitialConditions(E, n, m, alg::Val{:poincare_uniform},
        symmetric=Val(true); param=(A=1, B=0.55, D=0.4), border_n=1000)
    border, ex, ey = phase_space_border(E, symmetric, border_n, param=param)
    counter = 0
    points = Vector{Point2D}()
    it_grid = Iterators.product(range(1,stop=2-1e-14,length=n),
        range(1,stop=2-1e-14,length=m))
    grid = [Point(x, y) for (x, y) ∈ it_grid]
    for p in grid
        inpolygon(border, p) && (push!(points, p); counter += 1)
    end
    @debug "Generated $count initial conditions."
    frominterval(points, ex, ey)
end

end  # module InitialConditions
