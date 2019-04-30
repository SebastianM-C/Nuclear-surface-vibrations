module Visualizations

export poincare_explorer, animate, animate_solution, θϕ_sections,
    scene_limits2D, scene_limits3D, path_animation2D, path_animation3D,
    plot_slice!, parallel_paths, paths_distance

using ..Hamiltonian
using ..InitialConditions
using ..InitialConditions: initial_conditions!
using ..Poincare
using ..Lyapunov
using ..Lyapunov: λmap!
using ..DInfty
using ..DInfty: d∞!
using ..ParallelTrajectories
using ..DataBaseInterface

using AbstractPlotting, GLMakie
using AbstractPlotting: interpolated_getindex
using StatsBase
using StatsMakie
using Statistics
using IntervalArithmetic
using StaticArrays
using OrdinaryDiffEq
using RecursiveArrayTools
using DiffEqBase
using Colors: color
using LinearAlgebra: norm


x(r, θ, ϕ) = r*sin(θ)*cos(ϕ)
y(r, θ, ϕ) = r*sin(θ)*sin(ϕ)
z(r, θ, ϕ) = r*cos(θ)

function R(a₀, a₂, θ, ϕ; R₀=1)
    ξ = x(1,θ,ϕ)
    η = y(1,θ,ϕ)
    ζ = z(1,θ,ϕ)

    return R₀*(1 + √(5/4π)*(√(3/2)*((-a₀/√6 + a₂)*ξ^2 - (a₀/√6 + a₂)*η^2) + a₀*ζ^2))
end

R(sol::DiffEqBase.AbstractODESolution, t, θ, ϕ; R₀=1) = lift(t->R.(sol(t, idxs=3), sol(t, idxs=4)*√2, θ, ϕ, R₀=R₀), t)

function scene_limits3D(x, y, z)
    xm, xM = extrema(x)
    ym, yM = extrema(y)
    zm, zM = extrema(z)

    FRect3D((xm,ym,zm), (xM-xm,yM-ym,zM-zm))
end

function scene_limits3D(sol, idxs=[1,2,3])
    scene_limits3D(sol[idxs[1],:], sol[idxs[2],:], sol[idxs[3],:])
end

function scene_limits2D(x, y)
    xm, xM = extrema(x)
    ym, yM = extrema(y)

    FRect2D((xm,ym), (xM-xm,yM-ym))
end

function scene_limits2D(sol; idxs=[1,2])
    scene_limits2D(sol[idxs[1],:], sol[idxs[2],:])
end

function plot_sim(sim; colors=float.(axes(sim, 1)), idxs=[1,2])
    ui, ms = AbstractPlotting.textslider(range(0.001, stop=1., length=1000), "scale", start=0.05)
    data = Scene(resolution=(1000, 1000))
    colormap = to_colormap(:inferno, size(sim, 1))
    get_color(i) = interpolated_getindex(colormap, colors[i], extrema(colors))
    series_alpha = map(eachindex(sim)) do i
        simᵢ = sim[i]
        alpha = Node(1.0)
        if length(sim[i]) ≠ 0
            cmap = lift(α-> RGBAf0.(color.(fill(get_color(i), size(simᵢ, 1))), α), alpha)
            scatter!(data, [Point2f0(simᵢ[i][idxs[1]], simᵢ[i][idxs[2]]) for i in axes(simᵢ)[1]],
            colormap=cmap, color=fill(colors[i], size(simᵢ, 1)), markersize=ms)
        end
        alpha
    end

    scene = Scene()

    hbox(ui, data, parent=scene)
    return scene, series_alpha
end

function plot_hist(hist)
    cmap = to_colormap(:viridis, length(hist.weights))
    hist_α = [Node(1.) for i in cmap]
    bincolor(αs...) = RGBAf0.(color.(cmap), αs)
    colors = lift(bincolor, hist_α...)
    hist_sc = plot(hist, color=colors)

    return hist_sc, hist_α
end

function change_α(series_alpha, idxs, α=0.001)
    foreach(i-> series_alpha[i][] = α, idxs)
end

function get_series_idx(selected_plot, scene)
    plot_idx = findfirst(map(p->selected_plot === p, scene.plots))
    # println("scatter ", plot_idx)

    plot_idx
end

function setup_click(scene, idx=1)
    selection = Node{Any}(0)
    on(scene.events.mousebuttons) do buttons
        if ispressed(scene, Mouse.left) && AbstractPlotting.is_mouseinside(scene)
            plt, click_idx = AbstractPlotting.mouse_selection(scene)
            selection[] = (plt, click_idx)[idx]
        end
    end
    return selection
end

bin_with_val(val, hist) = searchsortedfirst(hist.edges[1], val) - 1

function idxs_in_bin(i, hist, val)
    h = hist.edges[1]
    bin = interval(h[i], h[i + 1])
    idx = findall(x -> x ∈ bin, val)
    return idx
end

function select_series(scene, selected_plot, scatter_α, hist_α, data, hist)
    series_idx = map(get_series_idx, selected_plot, scene)
    on(series_idx) do i
        if !isa(i, Nothing)
            scatter_α[i - 1][] = 1.
            change_α(scatter_α, setdiff(axes(scatter_α, 1), i - 1))
            selected_bin = bin_with_val(data[i-1], hist)
            hist_α[selected_bin][] = 1.
            change_α(hist_α, setdiff(axes(hist_α, 1), selected_bin))
        else
            change_α(scatter_α, axes(scatter_α, 1), 1.)
            change_α(hist_α, axes(hist_α, 1), 1.)
        end
        return nothing
    end
end

function select_bin(hist_idx, hist, hist_α, scatter_α, data)
    on(hist_idx) do i
        if i ≠ 0
            hist_α[i][] = 1.
            change_α(hist_α, setdiff(axes(hist.weights, 1), i))
            change_α(scatter_α, idxs_in_bin(i, hist, data), 1.)
            change_α(scatter_α, setdiff(axes(scatter_α, 1), idxs_in_bin(i, hist, data)))
        else
            change_α(scatter_α, axes(scatter_α, 1), 1.)
            change_α(hist_α, axes(hist_α, 1), 1.)
        end
        return nothing
    end
end

function poincare_explorer(g, E, f, alg, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, t=alg.T, rootkw=(xrtol=1e-6, atol=1e-6))
    # g, t_ = @timed initialize()
    # @debug "Loading graph took $t_ seconds."
    q0, p0 = initial_conditions!(g, E, alg=ic_alg, params=params)

    sim, t_ = @timed poincaremap(q0, p0, params=params, sgn=sgn, axis=axis, t=t, rootkw=rootkw)
    @debug "Poincaré map took $t_ seconds."

    ic_dep = InitialConditions.depchain(params, E, ic_alg)
    vals, t_ = @timed f(g, E, params=params, ic_alg=ic_alg, alg=alg)
    @debug "Loading or computing values took $t_ seconds."
    hist = fit(StatsBase.Histogram, vals, nbins=nbins, closed=:left)

    colors = Float32.(axes(hist.weights, 1))

    scatter_sc_with_ui, scatter_α = plot_sim(sim, colors=vals)
    scatter_sc = scatter_sc_with_ui.children[2]

    hist_sc, hist_α = plot_hist(hist)

    if axis == 3
        scatter_sc[Axis][:names, :axisnames][] = ("q₂","p₂")
    else
        scatter_sc[Axis][:names, :axisnames][] = ("q₁","p₁")
    end
    sc = AbstractPlotting.vbox(scatter_sc_with_ui, hist_sc)

    selected_plot = setup_click(scatter_sc, 1)
    hist_idx = setup_click(hist_sc, 2)

    select_series(scatter_sc, selected_plot, scatter_α, hist_α, vals, hist)
    select_bin(hist_idx, hist, hist_α, scatter_α, vals)

    return sc
end

function poincare_explorer(g, E, alg::LyapunovAlgorithm, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, rootkw=(xrtol=1e-6, atol=1e-6))
    scene = poincare_explorer(g, E, λmap!, alg, ic_alg; params=params, axis=axis, sgn=sgn,
        nbins=nbins, rootkw=rootkw)
    scene.children[2][Axis][:names, :axisnames][] = ("λ", "N")
    return scene
end

function poincare_explorer(g, E, alg::DInftyAlgorithm, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, rootkw=(xrtol=1e-6, atol=1e-6))
    poincare_explorer(g, E, d∞!, (d∞_alg = alg,), ic_alg; params=params, axis=axis, sgn=sgn,
        nbins=nbins, rootkw=rootkw)
end

function plot_err(errs)
    scene = Scene()
    colors = to_colormap(:viridis, size(errs, 1))

    series_alpha = map(eachindex(errs)) do i
        alpha = Node(1.0)
        if length(errs[i]) ≠ 0
            c = lift(α-> RGBAf0.(color.(fill(colors[i])), α), alpha)
            lines!(scene, errs[i], color=c)
        end
        alpha
    end
    scene[Axis][:names, :axisnames] = ("Intersection index", "Energy error")

    return scene, series_alpha
end

function poincare_error(E, ic_alg, t=1e4; params=PhysicalParameters(), axis=3,
        sgn=-1, rootkw=(xrtol=1e-6, atol=1e-6))
    q0, p0 = initial_conditions(E, alg=ic_alg)
    sim = poincaremap(q0, p0, params=params, sgn=sgn, axis=axis, t=t,
        rootkw=rootkw, full=true)
    err(z) = [H(zᵢ[SVector{2}(1:2)], zᵢ[SVector{2}(3:4)], params) - E for zᵢ in z]
    errs = err.(sim)

    err_sc, err_α = plot_err(errs)
    idxs = axis == 3 ? (2,4) : (1,3)
    scatter_sc_with_ui, scatter_α = plot_sim(sim, colors=maximum.(errs), idxs=idxs)
    scatter_sc = scatter_sc_with_ui.children[2]
    if axis == 3
        scatter_sc[Axis][:names, :axisnames] = ("q₂","p₂")
    else
        scatter_sc[Axis][:names, :axisnames] = ("q₁","p₁")
    end

    selected_plot = setup_click(scatter_sc, 1)
    # err_idx = setup_click(err_sc, 1)

    series_idx = map(get_series_idx, selected_plot, scatter_sc)
    on(series_idx) do i
        if !isa(i, Nothing)
            scatter_α[i - 1][] = 1.
            change_α(scatter_α, setdiff(axes(scatter_α, 1), i - 1))
            err_α[i - 1][] = 1.
            change_α(err_α, setdiff(axes(err_α, 1), i - 1))
        else
            change_α(scatter_α, axes(scatter_α, 1), 1.)
            change_α(err_α, axes(err_α, 1), 1.)
        end
        return nothing
    end

    scene = Scene()
    AbstractPlotting.vbox(scatter_sc_with_ui, err_sc, parent=scene)
end

function grids(N=100; θ_range=(0,π), ϕ_range=(0,2π))
    θ = range(θ_range..., length=N)
    ϕ = range(ϕ_range..., length=2N)
    Θ = [ϑ for ϑ in θ, φ in ϕ]
    Φ = [φ for ϑ in θ, φ in ϕ]

    return Θ, Φ
end

function animate_solution(sol, t; θ_range=(0,π), ϕ_range=(0,2π), R₀=1)
    θ, ϕ = grids(θ_range=θ_range, ϕ_range=ϕ_range)

    r = R(sol, t, θ, ϕ)
    X = lift(r->x.(r,θ,ϕ), r)
    Y = lift(r->y.(r,θ,ϕ), r)
    Z = lift(r->z.(r,θ,ϕ), r)

    q₀(t) = sol(t, idxs=3)
    q₂(t) = sol(t, idxs=4)
    all_r = [R.(q₀(t), q₂(t), θ, ϕ, R₀=R₀) for t ∈ sol.t]
    ex = convert(Matrix, VectorOfArray([[extrema(x.(r,θ,ϕ))...] for r in all_r]))
    ey = convert(Matrix, VectorOfArray([[extrema(y.(r,θ,ϕ))...] for r in all_r]))
    ez = convert(Matrix, VectorOfArray([[extrema(z.(r,θ,ϕ))...] for r in all_r]))
    limits = scene_limits3D(ex, ey, ez)

    return surface(X, Y, Z, limits=limits, colormap=:inferno, scale_plot=false)
end

function animate(t, tspan, Δt=0.02)
    for tᵢ in range(tspan..., step=Δt)
        push!(t, tᵢ)
        sleep(1/120)
    end
end

function θ_section(sol, t, N=100; θ=π/2, ϕ=range(0, 2π, length=N), limits, R₀=1)
    r = R(sol, t, θ, ϕ)

    X = lift(r->x.(r,θ,ϕ), r)
    Y = lift(r->y.(r,θ,ϕ), r)

    return lines(X, Y, axis=(names=(axisnames=("x","y"),),), limits=limits, scale_plot=false)
end

function ϕ_section(sol, t, N=100; θ=range(-π, π, length=N), ϕ=0, limits, R₀=1)
    r = R(sol, t, θ, ϕ)
    X = lift(r->x.(r,θ,ϕ), r)
    Z = lift(r->z.(r,θ,ϕ), r)

    return lines(X, Z, axis=(names=(axisnames=("x","z"),),), limits=limits, scale_plot=false)
end

function θϕ_sections(sol, t, limits)
    xm, xw = limits.origin[1], limits.widths[1]
    ym, yw = minimum(limits.origin[2:3]), maximum(limits.widths[2:3])
    lim = FRect2D((xm,ym), (xw,yw))
    θ_sc = θ_section(sol, t, limits=lim)
    ϕ_sc = ϕ_section(sol, t, limits=lim)

    return hbox(θ_sc, ϕ_sc)
end

function path_animation2D(sol, t; idxs=[1,2])
    init = [Point2f0(sol(t[], idxs=idxs[1]), sol(t[], idxs=idxs[2]))]
    trajectory = lift(t->push!(trajectory[],
        Point2f0(sol(t, idxs=idxs[1]), sol(t, idxs=idxs[2]))), t;
        init=init)
    limits = scene_limits2D(sol, idxs=idxs)
    lines(trajectory, limits=limits, markersize=0.7, scale_plot=false)
end

function path3D(sol, t, idxs)
    init = [Point3f0(sol(t[], idxs=idxs[1]), sol(t[], idxs=idxs[2]), sol(t[], idxs=idxs[3]))]
    trajectory = lift(t->push!(trajectory[],
        Point3f0(sol(t, idxs=idxs[1]), sol(t, idxs=idxs[2]), sol(t, idxs=idxs[3]))), t;
        init=init)
end

function path_animation3D(sol, t; idxs=[3,4,2,1], labels=(axisnames=("q₀","q₂","p₂"),))
    trajectory = path3D(sol, t, idxs)

    clims = extrema(sol[idxs[4],:])
    colormap = to_colormap(:viridis, length(sol))
    cinit = [interpolated_getindex(colormap, sol(t[], idxs=idxs[4]), clims)]
    colors = lift(t->push!(colors[],
        interpolated_getindex(colormap, sol(t, idxs=idxs[4]), clims)), t;
        init=cinit)

    limits = scene_limits3D(sol, idxs)
    sc = lines(trajectory, limits=limits, scale_plot=false, markersize=0.7,
        color=colors, colormap=colormap, axis=(names=labels,))

    o = [limits.origin...,]
    o[1] = 0
    w = [limits.widths...,]
    w[1] = 0.002
    intersection_plane = FRect3D(o, w)
    mesh!(sc, intersection_plane, color=(:blue, 0.2), limits=limits)
end

function plot_slice!(scene, sim; idxs=[1,2])
    meshscatter!(scene, [Point3f0(0, sim[i][idxs]...) for i in axes(sim,1)],
        markersize=0.015, limits=scene.limits, color=axes(sim,1), scale_plot=false,
        colormap=:inferno)
end

function parallel_paths(sol, t, idxs=[3,4,2,1,7,8,6,5], labels=(axisnames=("q₀","q₂","p₂"),))
    trajectory1 = path3D(sol, t, idxs[1:3])
    trajectory2 = path3D(sol, t, idxs[5:7])
    limits = scene_limits3D(sol, idxs)
    sc = lines(trajectory1, limits=limits, scale_plot=false, markersize=0.7, axis=(names=labels,))
    lines!(sc, trajectory2, limits=limits, scale_plot=false, markersize=0.7, axis=(names=labels,))
end

function log_ticks(lims, l)
    a = log10(lims[1])
    b = log10(lims[2])
    10 .^range(a, b, length=l)
end

function paths_distance(sol, t)
    idx1 = SVector{4}(1:4)
    idx2 = SVector{4}(5:8)

    @inbounds dist(u) = norm(u[idx1] - u[idx2])

    init = [Point2f0(t[], dist(sol(t[])))]
    distance = lift(t->push!(distance[], Point2f0(t, dist(sol(t)))),t; init=init)
    ey = extrema(dist.(sol[:]))
    limits = Node(FRect2D((0,ey[1]), (sol.t[end], ey[2]-ey[1])))
    ranges = Node((range(sol.t[1], sol.t[end], length=11), log_ticks(ey, 5)))
    # ranges = Node(([0,10,50,100], [1e-10, 1e-9, 1e-8]))
    # labels = Node((string.([0,10,50,100]), string.([1e-10, 1e-9, 1e-8])))
    
    lines(distance,
        # axis = (
        #     ticks = (
        #         ranges = ranges,
        #         labels = labels
        #     ),
        # ),
        limits = limits,
    )
end

end  # module Visualizations
