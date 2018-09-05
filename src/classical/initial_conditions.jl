module InitialConditions

export initial_conditions, update, InitialConditionsAlgorithm, Plane,
    PoincareRand, PoincareUniform, InscribedCircle, unpack_with_nothing

using ..Hamiltonian
using ..Parameters
using ..DataBaseInterface
using ..Classical: AbstractAlgorithm

using Logging, Random
using NLsolve, NLopt, Roots
using Plots, LaTeXStrings
using DataFrames, CSV
using GeometricalPredicates

abstract type InitialConditionsAlgorithm <: AbstractAlgorithm end
abstract type Plane end
struct Symmetric <: Plane end
struct Asymmetric <: Plane end

@with_kw struct PoincareRand{P <: Plane} <: InitialConditionsAlgorithm
    n::Int
    plane::P = Symmetric()
    border_n::Int = 1000
    @assert n > 0
    @assert border_n > 3
end

@with_kw struct PoincareUniform{P <: Plane} <: InitialConditionsAlgorithm
    n::Int
    m::Int
    plane::P = Symmetric()
    border_n::Int = 1000
    @assert n > 0
    @assert m > 0
    @assert border_n > 3
end

@with_kw struct InscribedCircle <: InitialConditionsAlgorithm
    n::Int
    m::Int
    @assert n > 0
    @assert m > 0
end

function unpack_with_nothing(alg::InitialConditionsAlgorithm)
    n = alg.n
    m = isa(alg, PoincareRand) ? nothing : alg.m
    border_n = isa(alg, InscribedCircle) ? nothing : alg.border_n

    return n, m, border_n
end

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

function initial_conditions(E, alg::InscribedCircle;
        params=PhysicalParameters())
    @unpack n, m = alg
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

function phase_space_border(E, plane::Symmetric, n=1000; params=PhysicalParameters())
    p = range(0, stop=√(2 * params.A * E), length=n)
    q0 = E < 1e5 ? 0 : 10
    q = [find_zero(q->H([0,pᵢ], [0,q], params) - E, q0) for pᵢ in p]
    points = Point.(q, p) ∪ Point.(-q[end-1:-1:1], p[end-1:-1:1]) ∪
        Point.(-q[2:end], -p[2:end]) ∪ Point.(q[end-1:-1:2], -p[end-1:-1:2])
    ex = extrema(p._x for p in points)
    ey = extrema(p._y for p in points)
    Polygon(tointerval(points, ex, ey)...), ex, ey
end

function phase_space_border(E, plane::Asymmetric, n=1000; params=PhysicalParameters())
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

function complete(q₂, p₂, E, plane::Symmetric, params)
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

function complete(q₀, p₀, E, plane::Asymmetric, params)
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

function initial_conditions(E, alg::PoincareRand;
        params=PhysicalParameters())
    @unpack n, plane, border_n = alg
    border, ex, ey = phase_space_border(E, plane, border_n, params=params)
    counter = 0
    points = Vector{Point2D}()
    while counter < n
        p = rand(Point2D)
        inpolygon(border, p) && (push!(points, p); counter += 1)
    end
    complete(frominterval(points, ex, ey)..., E, plane, params)
end

function initial_conditions(E, alg::PoincareUniform;
        params=PhysicalParameters())
    @unpack n, m, plane, border_n = alg
    border, ex, ey = phase_space_border(E, plane, border_n, params=params)
    counter = 0
    points = Vector{Point2D}()
    it_grid = Iterators.product(range(1, stop=2-1e-14, length=n),
        range(1,stop=2-1e-14,length=m))
    grid = [Point(x, y) for (x, y) ∈ it_grid]
    for p in grid
        inpolygon(border, p) && (push!(points, p); counter += 1)
    end
    @debug "Generated $counter initial conditions."
    counter == 0 && throw(ErrorException("No initial conditions generated!"))
    complete(frominterval(points, ex, ey)..., E, plane, params)
end

function poincare_err(q, p, err, plane::Symmetric)
    scatter(q[:, 2], p[:, 2], marker_z=err, m=1.2, msa=0,
        label="", xlabel=L"q_2", ylabel=L"p_2")
end

function poincare_err(q, p, err, plane::Asymmetric)
    scatter(q[:, 1], p[:, 1], marker_z=err, m=1.2, msa=0,
        label="", xlabel=L"q_0", ylabel=L"p_0")
end

function energy_err(q, p, E, alg, params)
    err = abs.([H(p[i,:], q[i,:], params) - E for i ∈ axes(q, 1)])
    plt = histogram(err, nbins=10, xlabel=L"\Delta E", ylabel=L"N",
        label="max err: $(maximum(err))")
    maximum(err) > 1e-12 && @warn "max err: $(maximum(err))"
    if !isa(alg, InscribedCircle)
        plt = plot(plt, poincare_err(q, p, err, alg.plane), layout=(2,1))
    else
        plt2 = scatter(q[:, 1], q[:, 2], marker_z=err, m=1.2, msa=0,
            label="", xlabel=L"q_0", ylabel=L"q_2")
        plt = plot(plt, plt2)
    end

    return plt
end

function save_err(plt, alg, prefix)
    fn = string(typeof(alg))
    fn = replace(fn, "NuclearSurfaceVibrations.Classical.InitialConditions." => "")
    if !isdir("$prefix/initial_energy_err")
        mkpath("$prefix/initial_energy_err")
    end
    savefig(plt, "$prefix/initial_energy_err/$fn.pdf")
end

function build_df(q, p, E, alg)
    N = size(q, 1)
    n, m, border_n = unpack_with_nothing(alg)
    df = DataFrame()
    df[:q₀] = categorical(q[:,1])
    df[:q₂] = categorical(q[:,2])
    df[:p₀] = categorical(p[:,1])
    df[:p₂] = categorical(p[:,2])
    df[:n] = categorical(fill(n, N))
    df[:m] = categorical((fill(m, N)))
    df[:E] = categorical(fill(E, N))
    df[:initial_cond_alg] = categorical(fill(string(typeof(alg)), N))
    df[:border_n] = categorical((fill(border_n, N)))
    allowmissing!(df)

    return df
end

function DataBaseInterface.DataBase(E, params=PhysicalParameters())
    col_names = ["q₀", "q₂", "p₀", "p₂", "n", "m", "E", "initial_cond_alg", "border_n"]
    types = [Union{Missing, Float64}, # q₀
            Union{Missing, Float64}, # q₂
            Union{Missing, Float64}, # p₀
            Union{Missing, Float64}, # p₂
            Union{Missing, Int64}, # n
            Union{Missing, Int64}, # m
            Union{Missing, Float64}, # E
            Union{Missing, String}, # initial_cond_alg
            Union{Missing, Int64} # border_n
            ]
    columns = Dict(col_names .=> types)
    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    location = (prefix, "z0.csv")
    # check for additional columns
    all_cols = split(readline(joinpath(location...)), ",")
    if length(all_cols) > length(col_names)
        standard_col_idx = findfirst.(isequal.(col_names, Ref(all_cols)))
        others = all_cols[setdiff(axes(all_cols, 1), standard_col_idx)]
        @debug "other cols" others
        # register types
        if !isa(findfirst(isequal("λs"), others), Nothing)
            push!(columns, "λs"=>Union{Missing, Float64})
        elseif !isa(findfirst(isequal("lyap_alg"), others), Nothing)
            push!(columns, "lyap_alg"=>Union{Missing, String})
        elseif !isa(findfirst(isequal("lyap_T"), others), Nothing)
            push!(columns, "lyap_T"=>Union{Missing, Float64})
        elseif !isa(findfirst(isequal("lyap_Ttr"), others), Nothing)
            push!(columns, "lyap_Ttr"=>Union{Missing, Float64})
        elseif !isa(findfirst(isequal("lyap_d0"), others), Nothing)
            push!(columns, "lyap_d0"=>Union{Missing, Float64})
        elseif !isa(findfirst(isequal("lyap_ut"), others), Nothing)
            push!(columns, "lyap_ut"=>Union{Missing, Float64})
        elseif !isa(findfirst(isequal("lyap_lt"), others), Nothing)
            push!(columns, "lyap_lt"=>Union{Missing, Float64})
        elseif !isa(findfirst(isequal("lyap_dt"), others), Nothing)
            push!(columns, "lyap_dt"=>Union{Missing, Float64})
        elseif !isa(findfirst(isequal("lyap_integ"), others), Nothing)
            push!(columns, "lyap_integ"=>Union{Missing, String})
        elseif !isa(findfirst(isequal("lyap_diffeq_kw"), others), Nothing)
            push!(columns, "lyap_diffeq_kw"=>Union{Missing, String})
        end
        length(all_cols) ≠ length(columns) && ErrorException("Unknown columns found in $others")
    end
    DataBase(location, columns)
end

function DataBaseInterface.update!(db::DataBase, df, ic_cond, vals)
    DataBaseInterface.fix_column_types(db, df)
    icdf = db.df[ic_cond, names(df)]

    cond = compatible(icdf, vals)
    @debug "update" cond
    update!(icdf, df, cond)
    @debug "done"
    for c in names(icdf)
        db.df[c][ic_cond] .= icdf[c]
    end
    @debug "check update" db.df
    update_file!(db)
end

function initial_conditions(E; alg=PoincareRand(n=5000), params=PhysicalParameters(),
        recompute=false)
    prefix = "output/classical/B$(params.B)-D$(params.D)/E$E"
    if !isdir(prefix)
        mkpath(prefix)
    end

    n, m, border_n = unpack_with_nothing(alg)

    if !isfile("$prefix/z0.csv")
        @debug "No initial conditions file found. Generating new conditions."
        q, p = initial_conditions(E, alg; params=params)
        df = build_df(q, p, E, alg)

        db = DataBase((prefix, "z0.csv"), df)

        plt = energy_err(q, p, E, alg, params)
        save_err(plt, alg, prefix)
    else
        db = DataBase(E, params)
        vals = Dict([:n, :m, :E, :initial_cond_alg, :border_n] .=>
                    [n, m, E, string(typeof(alg)), border_n])
        cond = compatible(db.df, vals)
        @debug "compatible" cond

        compat = count(cond) > 0 && !recompute

        if compat
            unique_df = unique(view(db.df, cond))
            @debug "total size" size(db.df) size(unique_df)
            q = hcat(unique_df[:q₀], unique_df[:q₂])
            p = hcat(unique_df[:p₀], unique_df[:p₂])
        else
            @debug "Incompatible initial conditions. Generating new conditions."
            q, p = initial_conditions(E, alg; params=params)
            df = build_df(q, p, E, alg)

            if recompute
                DataBaseInterface.deleterows!(db, cond)
            end
            append_with_missing!(db, df)
            update_file!(db)
            plt = energy_err(q, p, E, alg, params)
            save_err(plt, alg, prefix)
        end
    end
    arr_type = nonnothingtype(eltype(q))
    return Array{arr_type}(disallowmissing(q)), Array{arr_type}(disallowmissing(p))
end

end  # module InitialConditions
