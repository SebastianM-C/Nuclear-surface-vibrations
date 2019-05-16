module Visualizations

export poincare_explorer, animate, animate_solution, θϕ_sections,
    scene_limits2D, scene_limits3D, path_animation2D, path_animation3D,
    plot_slice!, parallel_paths, paths_distance, paths_distance_log,
    save_animation, selected_hist

using ..Hamiltonian
using ..InitialConditions
using ..InitialConditions: depchain, initial_conditions!
using ..Poincare
using ..Lyapunov
using ..Lyapunov: λmap!
using ..DInfty
using ..DInfty: d∞!
using ..ParallelTrajectories
using ..DataBaseInterface
using ..Reductions

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
using LightGraphs
using StorageGraphs
using UnicodeFun
using InteractiveChaos

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

function poincare_explorer(g, E, f, alg, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, t=alg.T, rootkw=(xrtol=1e-6, atol=1e-6),
        markersize=0.1)
    # g, t_ = @timed initialize()
    # @debug "Loading graph took $t_ seconds."
    ic_node = initial_conditions!(g, E, alg=ic_alg, params=params)
    ic_dep = depchain(params, E, ic_alg)
    outn = outneighbors(g, ic_node)
    q0 = ic_node[:q0]
    p0 = ic_node[:p0]

    if length(outn) > 0 && any(has_prop.(Ref(g), outn, :poincare_axis))
        # we have some Poincaré sections computed, we now have to check if they are the
        # right ones
        @debug "Looking for available Poincaré sections"
        idx = findfirst(v->g[v] == (poincare_axis=axis,sgn=sgn,t=t,rootkw=rootkw), outn)
        if idx !== nothing
            @debug "Found compatible Poincaré sections"
            vals, t_ = @timed g[:sim, ic_dep..., (poincare_axis=axis,sgn=sgn,t=t,rootkw=rootkw)]
            @assert length(vals) == 1 "Poincaré section not uniquely represented by deps!"
            sim = vals[1]
            @debug "Loading saved Poincaré sections took $t_ seconds."
        else
            sim, t_ = @timed poincaremap(q0, p0, params=params, sgn=sgn, axis=axis, t=t, rootkw=rootkw)
            @debug "Poincaré section took $t_ seconds."
            _, t_ = @timed add_nodes!(g, foldr(=>,(ic_dep..., ic_node,
                (poincare_axis=axis,sgn=sgn,t=t,rootkw=rootkw), (sim=sim,))))
            @debug "Adding Poincaré section to graph took $t_ seconds."
        end
    else
        sim, t_ = @timed poincaremap(q0, p0, params=params, sgn=sgn, axis=axis, t=t, rootkw=rootkw)
        @debug "Poincaré section took $t_ seconds."
        _, t_ = @timed add_nodes!(g, foldr(=>,(ic_dep..., ic_node,
            (poincare_axis=axis,sgn=sgn,t=t,rootkw=rootkw), (sim=sim,))))
        @debug "Adding Poincaré section to graph took $t_ seconds."
    end
    ds = Dataset{2, Float64}[]

    for s in sim
        push!(ds, Dataset{2, Float64}(s))
    end

    vals, t_ = @timed f(g, E, params=params, ic_alg=ic_alg, alg=alg)
    @debug "Loading or computing values took $t_ seconds."

    data_scene, hist_scene = trajectory_highlighter(ds, vals, nbins=50,
        cmap=:inferno, α=0.001, markersize = markersize)

    if axis == 3
        data_scene[Axis][:names, :axisnames][] = ("q₂","p₂")
    else
        data_scene[Axis][:names, :axisnames][] = ("q₁","p₁")
    end

    return data_scene, hist_scene
end

function poincare_explorer(g, E, alg::LyapunovAlgorithm, ic_alg;
        params=PhysicalParameters(), axis=3, sgn=-1, nbins=50, t=alg.T,
        rootkw=(xrtol=1e-6, atol=1e-6), markersize=0.1)
    data_scene, hist_scene = poincare_explorer(g, E, λmap!, alg, ic_alg;
        params=params, axis=axis, sgn=sgn, nbins=nbins, t=t, rootkw=rootkw,
        markersize=markersize)
    hist_scene[Axis][:names, :axisnames][] = ("λ", "N")
    return AbstractPlotting.vbox(data_scene, hist_scene)
end

function poincare_explorer(g, E, alg::DInftyAlgorithm, ic_alg;
        params=PhysicalParameters(), axis=3, sgn=-1, nbins=50, t=alg.T,
        rootkw=(xrtol=1e-6, atol=1e-6), markersize=0.1)
    data_scene, hist_scene = poincare_explorer(g, E, d∞!, alg, ic_alg;
        params=params, axis=axis, sgn=sgn, nbins=nbins, t=t, rootkw=rootkw,
        markersize=markersize)
    hist_scene[Axis][:names, :axisnames][] = ("d∞", "N")
    return AbstractPlotting.vbox(data_scene, hist_scene)
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

# function poincare_error(E, ic_alg, t=1e4; params=PhysicalParameters(), axis=3,
#         sgn=-1, rootkw=(xrtol=1e-6, atol=1e-6))
#     q0, p0 = initial_conditions(E, alg=ic_alg)
#     sim = poincaremap(q0, p0, params=params, sgn=sgn, axis=axis, t=t,
#         rootkw=rootkw, full=true)
#     err(z) = [H(zᵢ[SVector{2}(1:2)], zᵢ[SVector{2}(3:4)], params) - E for zᵢ in z]
#     errs = err.(sim)
#
#     err_sc, err_α = plot_err(errs)
#     idxs = axis == 3 ? (2,4) : (1,3)
#     scatter_sc_with_ui, scatter_α = plot_sim(sim, colors=maximum.(errs), idxs=idxs)
#     scatter_sc = scatter_sc_with_ui.children[2]
#     if axis == 3
#         scatter_sc[Axis][:names, :axisnames] = ("q₂","p₂")
#     else
#         scatter_sc[Axis][:names, :axisnames] = ("q₁","p₁")
#     end
#
#     selected_plot = setup_click(scatter_sc, 1)
#     # err_idx = setup_click(err_sc, 1)
#
#     series_idx = map(get_series_idx, selected_plot, scatter_sc)
#     on(series_idx) do i
#         if !isa(i, Nothing)
#             scatter_α[i - 1][] = 1.
#             change_α(scatter_α, setdiff(axes(scatter_α, 1), i - 1))
#             err_α[i - 1][] = 1.
#             change_α(err_α, setdiff(axes(err_α, 1), i - 1))
#         else
#             change_α(scatter_α, axes(scatter_α, 1), 1.)
#             change_α(err_α, axes(err_α, 1), 1.)
#         end
#         return nothing
#     end
#
#     scene = Scene()
#     AbstractPlotting.vbox(scatter_sc_with_ui, err_sc, parent=scene)
# end

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

    return surface(X, Y, Z,
        limits=limits,
        colormap=:solar,
        scale_plot=false)
end

function animate(t, tspan, Δt=0.02)
    for tᵢ in range(tspan..., step=Δt)
        push!(t, tᵢ)
        sleep(1/120)
    end
end

function save_animation(scene, t, tspan, fn, Δt=0.03)
    record(scene, fn, range(tspan..., step=Δt), framerate=60) do tᵢ
        push!(t, tᵢ)
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

function add_intersection_plane!(sc, limits=sc.limits[]; thickness=0.002, color=(:blue, 0.2))
    o = [limits.origin...,]
    o[1] = 0
    w = [limits.widths...,]
    w[1] = thickness
    intersection_plane = FRect3D(o, w)
    mesh!(sc, intersection_plane, color=color, limits=limits)
end

function path_animation3D(sol, t; idxs=[3,4,2,1], markersize=0.9,
        labels=(axisnames=("q₀","q₂","p₂"),))
    trajectory = path3D(sol, t, idxs)

    clims = extrema(sol[idxs[4],:])
    colormap = to_colormap(:viridis, length(sol))
    cinit = [interpolated_getindex(colormap, sol(t[], idxs=idxs[4]), clims)]
    colors = lift(t->push!(colors[],
        interpolated_getindex(colormap, sol(t, idxs=idxs[4]), clims)), t;
        init=cinit)

    limits = scene_limits3D(sol, idxs)
    sc = lines(trajectory, limits=limits, scale_plot=false, markersize=markersize,
        color=colors, colormap=colormap, axis=(names=labels,))

    add_intersection_plane!(sc, limits)
end

function plot_slice!(scene, sim; idxs=[1,2], markersize=0.015, scale_plot=false)
    meshscatter!(scene, [Point3f0(0, sim[i][idxs]...) for i in axes(sim,1)],
        markersize=markersize, limits=scene.limits, color=axes(sim,1),
        scale_plot=scale_plot, colormap=:inferno)
end

function endpoints!(scene, sol, t, idxs=[3,4,2,1,7,8,6,5]; color=:blue,
        markersize=0.01, linewidth=0.7)
    p = lift(t->[Point3f0(sol(t, idxs=idxs[1:3])),
                 Point3f0(sol(t, idxs=idxs[5:7]))], t)
    meshscatter!(scene, p, limits=scene.limits, markersize=markersize)
    lines!(scene, p, limits=scene.limits, color=color, linewidth=linewidth)
end

function parallel_paths(sol, t, idxs=[3,4,2,1,7,8,6,5],
        labels=(axisnames=("q₀","q₂","p₂"),); c1=:black, c2=:black, linecolor=:blue,
        tlw=0.7, lms=0.01, lw=2, scale_plot=false)
    trajectory1 = path3D(sol, t, idxs[1:3])
    trajectory2 = path3D(sol, t, idxs[5:7])
    limits = scene_limits3D(sol, idxs)

    sc = lines(trajectory1, limits=limits, scale_plot=scale_plot, linewidth=tlw,
        axis=(names=labels,), color=c1)
    lines!(sc, trajectory2, limits=limits, scale_plot=scale_plot, linewidth=tlw,
        axis=(names=labels,), color=c2)
    endpoints!(sc, sol, t, idxs, color=linecolor, markersize=lms, linewidth=lw)
end

function log_ticks(lims, n)
    a = round(lims[1], RoundNearest)
    b = round(lims[2], RoundNearest)
    r = range(a, b, length=n)
    l = raw"10^{".*string.(r).*"}"
    t = to_latex.(l)
    return r, t
end

function paths_distance_log(sol, t)
    idx1 = SVector{4}(1:4)
    idx2 = SVector{4}(5:8)

    @inbounds dist(u) = log10(norm(u[idx1] - u[idx2]))

    init = [Point2f0(t[], dist(sol(t[])))]
    distance = lift(t->push!(distance[], Point2f0(t, dist(sol(t)))),t; init=init)
    ey = extrema(dist.(sol[:]))
    limits = Node(FRect2D((0,ey[1]), (sol.t[end], ey[2]-ey[1])))
    r, t = log_ticks(ey, 5)
    ranges = Node((range(sol.t[1], sol.t[end], length=5), r))
    labels = lift(x->(string.(x[1]), t), ranges)

    lines(distance,
        axis = (
            names = (axisnames=("t", "log₁₀(d)"),),
            ticks = (
                ranges = ranges,
                labels = labels
            ),
        ),
        limits = limits,
    )
end

function paths_distance(sol, t; xlims=(sol.t[1], sol.t[end]), kwargs...)
    idx1 = SVector{4}(1:4)
    idx2 = SVector{4}(5:8)

    @inbounds dist(u) = norm(u[idx1] - u[idx2])

    init = [Point2f0(t[], dist(sol(t[])))]
    distance = lift(t->push!(distance[], Point2f0(t, dist(sol(t)))),t; init=init)
    ey = extrema(dist.(sol[:]))
    limits = Node(FRect2D((xlims[1],ey[1]), (xlims[2], ey[2]-ey[1])))

    lines(distance, limits = limits, axis=(names=(axisnames=("t","d"),),); kwargs...)
end

function selected_hist(g, E, f, alg, ic_alg; params,
        nbins=50, select=Reductions.select_after_first_max)
    ic_dep = depchain(params, E, ic_alg)
    vals = f(g, E, params=params, ic_alg=ic_alg, alg=alg)
    selected = select(vals)

    hist = fit(Histogram, vals, nbins=nbins, closed=:left)
    shist = fit(Histogram, selected, hist.edges[1], closed=:left)

    return hist, shist
end

function selected_hist(g, E, alg::LyapunovAlgorithm, ic_alg;
        params, nbins=50, select=Reductions.select_after_first_max)
    selected_hist(g, E, λmap!, alg, ic_alg, params=params, nbins=nbins, select=select)
end

function selected_hist(g, E, alg::DInftyAlgorithm, ic_alg;
        params, nbins=50, select=Reductions.select_after_first_max)
    selected_hist(g, E, d∞!, alg, ic_alg, params=params, nbins=nbins, select=select)
end

end  # module Visualizations
