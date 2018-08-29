#!/usr/bin/env julia
module InitialConditions

export initial_conditions

using Logging, Random
using NLsolve, NLopt, Roots
using Plots, LaTeXStrings
using DataFrames, CSV
using GeometricalPredicates

include("hamiltonian.jl")
using .Hamiltonian

function objective(q::Vector, grad::Vector)
    sqrt.(sum(q.^2))
end

function generate_inscribed(E, n, params)

    function constraint(q::Vector, grad::Vector)
        V(q, params) - E
    end

    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [-Inf, -Inf])

    xtol_rel!(opt, 1e-13)

    min_objective!(opt, objective)
    equality_constraint!(opt, constraint, 1e-13)

    r, q_min, ret_code = optimize(opt, [0., 0.])

    θ = range(0+0.1, stop=2π+0.1, length=n)

    x = r.*cos.(θ)
    y = r.*sin.(θ)

    hcat(x, y)
end

function find_qs(E, n, params)

    function f!(F, q)
        F[1] = V(q, params) - E
        F[2] = 0.
    end

    q0 = zeros(n, 2)
    qs = generate_inscribed(E, n, params)

    for i=1:n
        result = nlsolve(f!, qs[i,:], method=:trust_region, ftol=1e-13)
        !converged(result) && @debug "Did not converge for $i" result
        q0[i,:] = converged(result) ? result.zero : NaN
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
    N == 0 && throw(ErrorException("All initial conditions are invalid!"))

    return q0, p0
end

"""
    p2(T, p, A)

Compute ``p_2`` (or ``p_0``) when the other is given at kinetic energy `T` and 0 potential energy.
"""
p2(T, p, A) = √(2 * A * T - p^2)

"""
    p2(E, p, q, params)

Compute ``p_2`` (or ``p_0``) when the other is given at energy `E`.
"""
p2(E, p, q, params) = √(2 * params.A * (E - V(q, params)) - p^2)

function _initial_conditions(E, n, m, alg::Val{:inscribed_circle};
        params=(A=1, B=0.55, D=0.4))
    T_range = range(0, stop=E, length=m)

    q0 = zeros(n*m, 2)
    p0 = zeros(n*m, 2)

    for i ∈ Base.OneTo(m)
        idx = n*(i-1)+1:n*(i-1)+n
        p0[idx, 1] = range(0, stop=√(2 * params.A * T_range[i]), length=n)
        p0[idx, 2] = real(p2.(T_range[i]+0im, p0[idx, 1], params.A))

        q0[idx, :] = find_qs(E - T_range[i], n, params)
    end

    filter_NaNs!(q0, p0, n, m)
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

function phase_space_border(E, symmetric::Val{true}, n=1000; params=(A=1, B=0.55, D=0.4))
    p = range(0, stop=√(2 * params.A * E), length=n)
    q0 = E < 1e5 ? 0 : 10
    q = [find_zero(q->H([0,pᵢ], [0,q], params) - E, q0) for pᵢ in p]
    points = Point.(q, p) ∪ Point.(-q[end-1:-1:1], p[end-1:-1:1]) ∪
        Point.(-q[2:end], -p[2:end]) ∪ Point.(q[end-1:-1:2], -p[end-1:-1:2])
    ex = extrema(p._x for p in points)
    ey = extrema(p._y for p in points)
    Polygon(tointerval(points, ex, ey)...), ex, ey
end

function phase_space_border(E, symmetric::Val{false}, n=1000; params=(A=1, B=0.55, D=0.4))
    p = range(0, stop=√(2 * params.A * E), length=n)
    p = p[1:end-1]
    q₊0 = E < 1e5 ? (0,100) : (0,10000)
    q₋0 = E < 1e5 ? (-100,0) : (-10000,0)
    q₊ = [find_zero(q->H([pᵢ,0], [q,0], params) - E, q₊0) for pᵢ in p]
    q₋ = [find_zero(q->H([pᵢ,0], [q,0], params) - E, q₋0) for pᵢ in p]
    points = Point.(q₊, p) ∪ Point.(q₋[end-1:-1:1], p[end-1:-1:1]) ∪
        Point.(q₋[2:end], -p[2:end]) ∪ Point.(q₊[end-1:-1:2], -p[end-1:-1:2])
    ex = extrema(p._x for p in points)
    ey = extrema(p._y for p in points)
    Polygon(tointerval(points, ex, ey)...), ex, ey
end

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Point2D}) = Point((rand(rng) + 1), (rand(rng) + 1))

function complete(q₂, p₂, E, symmetric::Val{true}, params)
    n = length(q₂)
    q = zeros(eltype(q₂), n, 2)
    p = zeros(eltype(p₂), n, 2)
    q[:, 2] = q₂
    p[:, 2] = p₂
    for i ∈ axes(q, 1)
        p[i, 1] = real(p2(E+0im, p₂[i], q[i,:], params))
    end

    q, p
end

function complete(q₀, p₀, E, symmetric::Val{false}, params)
    n = length(q₀)
    q = zeros(eltype(q₀), n, 2)
    p = zeros(eltype(p₀), n, 2)
    q[:, 1] = q₀
    p[:, 1] = p₀
    for i ∈ axes(q, 1)
        p[i, 2] = real(p2(E+0im, p₀[i], q[i,:], params))
    end

    q, p
end

function _initial_conditions(E, n, alg::Val{:poincare_rand}, symmetric=Val(true);
        params=(A=1, B=0.55, D=0.4), border_n=1000)
    border, ex, ey = phase_space_border(E, symmetric, border_n, params=params)
    counter = 0
    points = Vector{Point2D}()
    while counter < n
        p = rand(Point2D)
        inpolygon(border, p) && (push!(points, p); counter += 1)
    end
    complete(frominterval(points, ex, ey)..., E, symmetric, params)
end

function _initial_conditions(E, n, m, alg::Val{:poincare_uniform}, symmetric=Val(true);
        params=(A=1, B=0.55, D=0.4), border_n=1000)
    border, ex, ey = phase_space_border(E, symmetric, border_n, params=params)
    counter = 0
    points = Vector{Point2D}()
    it_grid = Iterators.product(range(1,stop=2-1e-14,length=n),
        range(1,stop=2-1e-14,length=m))
    grid = [Point(x, y) for (x, y) ∈ it_grid]
    for p in grid
        inpolygon(border, p) && (push!(points, p); counter += 1)
    end
    @debug "Generated $counter initial conditions."
    counter == 0 && throw(ErrorException("No initial conditions generated!"))
    complete(frominterval(points, ex, ey)..., E, symmetric, params)
end

function poincare_err(q, p, err, symmetric::Val{true})
    scatter(q[:, 2], p[:, 2], marker_z=err, m=1.2, msa=0,
        label="", xlabel=L"q_2", ylabel=L"p_2")
end

function poincare_err(q, p, err, symmetric::Val{false})
    scatter(q[:, 1], p[:, 1], marker_z=err, m=1.2, msa=0,
        label="", xlabel=L"q_0", ylabel=L"p_0")
end

function energy_err(q, p, E, alg, symmetric, params)
    err = abs.([H(p[i,:], q[i,:], params) - E for i ∈ axes(q, 1)])
    plt = histogram(err, nbins=10, xlabel=L"\Delta E", ylabel=L"N",
        label="max err: $(maximum(err))")
    maximum(err) > 1e-12 && @warn "max err: $(maximum(err))"
    if !isa(alg, Val{:inscribed_circle})
        plt = plot(plt, poincare_err(q, p, err, symmetric), layout=(2,1))
    end

    return plt
end

"""
    val2str(v)

Extrct a string from a value type based on a symbol.
"""
val2str(v) = match(r":(\w+)", "$v").captures[1]

function save_err(plt, alg, symmetric, prefix)
    if isa(alg, Val{:inscribed_circle})
        fn = val2str(alg)
    elseif isa(symmetric, Val{true})
        fn = val2str(alg)*"_symm"
    else
        fn = val2str(alg)*"_asymm"
    end

    if !isdir("$prefix/initial_energy_err")
        mkpath("$prefix/initial_energy_err")
    end
    savefig(plt, "$prefix/initial_energy_err/$fn.pdf")
end

function build_df(q, p, m, n, E, alg, symmetric, border_n)
    N = size(q, 1)
    df = DataFrame()
    df[:q₀] = categorical(q[:,1])
    df[:q₂] = categorical(q[:,2])
    df[:p₀] = categorical(p[:,1])
    df[:p₂] = categorical(p[:,2])
    df[:n] = categorical(fill(n, N))
    df[:m] = categorical((fill(m, N)))
    df[:E] = categorical(fill(E, N))
    df[:initial_cond_alg] = categorical(fill("$alg", N))
    df[:symmetric] = categorical((fill(isa(symmetric, Missing) ? missing : "$symmetric", N)))
    df[:border_n] = categorical((fill(border_n, N)))
    allowmissing!(df)

    return df
end

function initial_conditions(E; n=5000, m=missing, params=(A=1, B=0.55, D=0.4),
        alg=Val(:poincare_rand), symmetric=Val(true), border_n=1000,
        recompute=false)
    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    if !isdir(prefix)
        mkpath(prefix)
    end

    if isa(alg, Val{:poincare_rand})
        n_ = n
        m = missing
    else
        n_ = (n, m)
    end
    if !isa(alg, Val{:inscribed_circle})
        alg_ = (alg, symmetric)
        params_ = Dict(:params=>params, :border_n=>border_n)
    else
        alg_ = (alg,)
        params_ = Dict(:params=>params)
        symmetric = missing
        border_n = missing
    end
    typeof(alg) <: Union{Val{:poincare_uniform}, Val{:inscribed_circle}} &&
        isa(m, Missing) && throw(ArgumentError("m not given"))

    if !isfile("$prefix/z0.csv")
        @debug "No initial conditions file found. Generating new conditions."
        q, p = _initial_conditions(E, n_..., alg_...; params_...)
        df = build_df(q, p, m, n, E, alg, symmetric, border_n)

        # TODO: Switch to JLD2 after https://github.com/simonster/JLD2.jl/issues/101 is fixed
        CSV.write("$prefix/z0.csv", df)

        plt = energy_err(q, p, E, alg, symmetric, params)
        save_err(plt, alg, symmetric, prefix)
    else
        df = CSV.read("$prefix/z0.csv", use_mmap=!Sys.iswindows())
        col_names = [:q₀, :q₂, :p₀, :p₂, :n, :m, :E, :initial_cond_alg, :symmetric, :border_n]
        any(.!haskey.(Ref(df), col_names)) && throw(ErrorException("Invalid DataFrame!\n$df"))
        # restore types
        for c in setdiff(names(df), [:m, :symmetric, :border_n])
            categorical!(df, c)
        end
        types = [Int, String, Int]
        for (i, c) in enumerate([:m, :symmetric, :border_n])
            df[c] = categorical(Array{Union{Missing,types[i]}}(df[c]))
        end
        # TODO: change comparison to value types after switching from CSV
        m_cond(v, x::Number) = isa(x, Missing) ? isa.(v, Missing) : v == x
        m_cond(v, x) = isa(x, Missing) ? isa.(v, Missing) : v == "$x"
        @debug "size" size(m_cond.(df[:symmetric], symmetric)) size(df[:E] .== E)
        cond = (df[:n] .== n) .& m_cond.(df[:m], m) .& (df[:E] .== E) .&
            (df[:initial_cond_alg] .== "$alg") .&
            m_cond.(df[:symmetric], symmetric) .& m_cond.(df[:border_n], border_n)
        @debug "m" m m_cond.(df[:border_n], border_n) cond
        filtered_df = df[cond[.!isa.(cond, Missing)], :]

        @debug "filter" size(filtered_df, 1)

        compatible = size(filtered_df, 1) > 0 && !recompute
        if compatible
            unique!(filtered_df)
            @debug "total size" size(df) size(filtered_df)
            q = hcat(filtered_df[:q₀], filtered_df[:q₂])
            p = hcat(filtered_df[:p₀], filtered_df[:p₂])
        else
            @debug "Incompatible initial conditions. Generating new conditions."
            q, p = _initial_conditions(E, n_..., alg_...; params_...)
            df_ = build_df(q, p, m, n, E, alg, symmetric, border_n)

            if recompute && size(filtered_df, 1) > 0
                # delete the old values
                @debug "Deleted" count(cond)
                deleterows!(df, axes(df[:n], 1)[cond])
            end
            # add the new values
            for c in setdiff(names(df), names(df_))
                df_[c] = categorical(fill(missing, size(df_, 1)))
            end
            append!(df, df_[names(df)])
            @debug "total size" size(df)
            CSV.write("$prefix/z0.csv", df)

            plt = energy_err(q, p, E, alg, symmetric, params)
            save_err(plt, alg, symmetric, prefix)
        end
    end
    return Array(disallowmissing(q)), Array(disallowmissing(p))
end

end  # module InitialConditions
