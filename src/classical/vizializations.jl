using Makie
using StatsBase
using StatsMakie
using Statistics
using IntervalArithmetic
using StaticArrays
using Colors: color

function plot_sim(sim; colors=axes(sim, 1), idxs=[1,2])
    ui, ms = AbstractPlotting.textslider(range(0.001, stop=1., length=1000), "scale", start=0.05)
    data = Scene(resolution=(1000, 1000))
    colormap = to_colormap(:viridis, size(sim, 1))
    get_color(i) = AbstractPlotting.interpolated_getindex(colormap, colors[i], extrema(colors))
    series_alpha = map(eachindex(sim)) do i
        simᵢ = sim[i]
        alpha = Node(1.0)
        if length(sim[i]) ≠ 0
            cmap = lift(α-> RGBAf0.(color.(fill(get_color(i), size(simᵢ, 1))), α), alpha)
            scatter!(data, [Point2f0(simᵢ[i, idxs[1]], simᵢ[i, idxs[2]]) for i in axes(simᵢ, 1)],
            colormap=cmap, color=fill(colors[i], size(simᵢ, 1)), markersize=ms)
        end
        alpha
    end

    scene = Scene()

    hbox(ui, data, parent=scene)
    return scene, series_alpha
end

function change_α(series_alpha, idxs, α=0.01)
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
            plt, click_idx = Makie.mouse_selection(scene)
            selection[] = (plt, click_idx)[idx]
        end
    end
    return selection
end

function select_series(scene, selected_plot, series_alpha, hist_sc, vals, hist)
    series_idx = map(get_series_idx, selected_plot, scene)
    on(series_idx) do i
        if !isa(series_idx[], Nothing)
            change_α(series_alpha, setdiff(axes(series_alpha, 1), series_idx[] - 1))
        else
            change_α(series_alpha, axes(series_alpha, 1), 1.)
        end
    end
end 

bin_with_val(val, hist) = searchsortedfirst(hist.edges[1], val) - 1

function idxs_in_bin(i, hist, val)
    h = hist.edges[1]
    bin = interval(h[i], h[i + 1])
    idx = findall(x -> x ∈ bin, val)
    return idx
end

function select_bin(scene, idx, l, hist, scatter_sc, v)
    original_sc_colors = copy(scatter_sc[end][:colormap][])
    original_hist_colors = copy(scene[end][:colormap][])
    # zero_sc = minimum(scatter_sc[end][:colorrange][])
    # zero_hist = minimum(scene[end][:colorrange][])

    on(idx) do click_idx
        println("histogram ", click_idx)
        if click_idx ≠ 0
            l_idx = idxs_in_bin(click_idx, hist, l)
            len = 1
            splot = scatter_sc[end]
            for i in axes(v, 1)
                if i ∉ l_idx
                    # splot[:color][][len:len+size(v[i], 1)-1] .= zero_sc
                    # splot[:color][] = splot[:color][]
                    change_colormap(splot[:colormap], i:i, 0.01)
                end
                len += size(v[i], 1)
            end
            # scene[end][:color][][setdiff(axes(scene[end][:color][], 1), click_idx)] .= zero_hist
            # scene[end][:color][] = scene[end][:color][]
            # change_colormap(scene[end][:colormap], setdiff(axes(scene[end][:colormap][], 1), click_idx))
        else
            println("outside")
            scatter_sc[end][:colormap][] = copy(original_sc_colors)
            scene[end][:colormap][] = copy(original_hist_colors)
        end
        return nothing
    end
end
# scatter_sc_with_ui.children[2][end][:colormap]
function poincare_explorer(E, alg, f, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, rootkw=(xrtol=1e-6, atol=1e-6))
    q0, p0 = initial_conditions(E, ic_alg)
    alg_type = typeof(alg)
    sim = poincaremap(q0, p0, params=params, sgn=sgn, axis=axis, t=alg.T, rootkw=rootkw)
    # f=λmap
    vals = f(E, alg=alg, ic_alg=ic_alg, params=params)
    hist = fit(StatsBase.Histogram, vals, nbins=nbins, closed=:left)

    colors = Float32.(axes(hist.weights, 1))
    hist_sc = plot(hist, color=colors, colormap=to_colormap(:viridis, length(colors)))
    scatter_sc_with_ui, series_alpha = plot_sim(sim, colors=vals)
    scatter_sc = scatter_sc_with_ui.children[2]
    if axis == 3
        scatter_sc[Axis][:names, :axisnames][] = ("q₂","p₂")
    else
        scatter_sc[Axis][:names, :axisnames][] = ("q₁","p₁")
    end
    sc = AbstractPlotting.vbox(scatter_sc_with_ui, hist_sc)

    selected_plot = setup_click(scatter_sc, 1)
    hist_idx = setup_click(hist_sc, 2)

    select_series(scatter_sc, selected_plot, series_alpha, hist_sc, vals, hist)
    # select_bin(hist_sc, hist_idx, l1, hist, scatter_sc, sim)

    return sc, selected_plot, series_alpha, hist_idx
end

function poincare_explorer(E, alg::LyapunovAlgorithm, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, rootkw=(xrtol=1e-6, atol=1e-6))
    scene = poincare_explorer(E, alg, λmap, ic_alg; params=params, axis=axis, sgn=sgn,
        nbins=nbins, rootkw=rootkw)
    scene.children[2][Axis][:names, :axisnames][] = ("λ", "N")
    return scene
end

function poincare_explorer(E, alg::DInftyAlgorithm, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, rootkw=(xrtol=1e-6, atol=1e-6))
    poincare_explorer(E, alg, d∞, ic_alg; params=params, axis=axis, sgn=sgn,
        nbins=nbins, rootkw=rootkw)
end

function plot_err(errs)
    scene = Scene(resolution=(1000,1000))
    points = Float32[]
    colors = Float32[]

    for i ∈ axes(errs, 1)
        eᵢ = errs[i]
        if length(eᵢ) ≠ 0
            append!(colors, Iterators.repeated(i, length(eᵢ)))
            append!(points, eᵢ)
        end
    end

    scatter!(scene, points, color=colors)
end

function poincare_error(E, ic_alg, T=1e4; params=PhysicalParameters(), axis=3,
        sgn=-1, rootkw=(xrtol=1e-6, atol=1e-6))
    q0, p0 = initial_conditions(E, alg=ic_alg)
    sim = poincaremap(q0, p0, params=params, sgn=sgn, axis=axis, t=T,
        rootkw=rootkw, full=true)
    err(z) = [H(zᵢ[SVector{2}(1:2)], zᵢ[SVector{2}(3:4)], params) - E for zᵢ in z]
    errs = err.(sim)

    err_sc = plot_err(errs)
    scatter_sc_with_ui = plot_sim(sim, colors=maximum.(errs), idxs=(2,4))
    scatter_sc = scatter_sc_with_ui.children[2]
    if axis == 3
        scatter_sc[Axis][:names, :axisnames] = ("q₂","p₂")
    else
        scatter_sc[Axis][:names, :axisnames] = ("q₁","p₁")
    end

    AbstractPlotting.vbox(scatter_sc_with_ui, err_sc)
end

E = 120.
# ic_alg = PoincareRand(n=200)
ic_alg = PoincareUniform(n=4,m=4)
alg = DynSys()

# s1 = poincare_explorer(E, alg, ic_alg)
# s2 = poincare_error(E, ic_alg, 1e4)

a = poincare_explorer(E, alg, λmap, ic_alg)
a[1]
a[2]
change_colormap(a[1].children[1].children[2][end][:colormap], 1:1, 1)
a[1].children[1].children[2][end][:color][]
### Random data
p1 = a[1].children[1].children[2].plots[2]
findfirst(map(p->p1===p, a[1].children[1].children[2].plots))

=== a[1].children[1].children[2].plots[3]
scene = Scene(resolution=(1000, 1000))
# ui, ms = AbstractPlotting.textslider(range(0.001, stop=1., length=1000), "scale", start=0.05)
sim = [rand(5,2), rand(7,2), rand(3,2)]
colors = rand(3)
# flatten data
series_alpha = map(axes(colors, 1)) do i
    simᵢ = sim[i]
    alpha = Node(1.0)
    if length(sim[i]) ≠ 0
        cmap = lift(a-> RGBAf0.(color.(to_colormap(:viridis, size(simᵢ, 1))), a), alpha)
        scatter!(scene, [Point2f0(simᵢ[i,1], simᵢ[i,2]) for i in axes(simᵢ, 1)],
            colormap=cmap, color=fill(colors[i], size(simᵢ, 1)),
            markersize=0.1)
    end
    alpha
end
scene
RGBAf0.(color.(to_colormap(:viridis)), 0.2)
scatter_sc = scatter(points, color=all_colors, colormap=to_colormap(:viridis, length(colors)))
Point2f0.(rand(4,3)[:,[1,2]])
click_idx = setup_click(scatter_sc)

series_idx = map(get_series_idx, click_idx, scatter_sc, sim)
on(series_idx) do i
    println("in series $i")
    if i ≠ 0
        change_colormap(scatter_sc[end][:colormap], setdiff(axes(colors, 1), i), 0.1)
    else
        change_colormap(scatter_sc[end][:colormap], axes(colors, 1), 1)
    end
end
scatter_sc[end][:colormap]

off(click_idx, click_idx.listeners[1])
off(series_idx, series_idx.listeners[1])
# , markersize=ms)
# hbox(ui, scatter_sc, parent=scene)
using Observables: off

change_colormap(scatter_sc[end][:colormap], 2:2, 0.4)

scene = Scene()
series_alpha = map(1:500) do i
  alpha = Node(1.0)
  cmap = lift(a-> RGBAf0.(color.(to_colormap(:viridis)), a), alpha)
  scatter!(rand(Point2f0, 1000) .+ (Point2f0(i, 0),), colormap = cmap, color = rand(1000))
  alpha
end
display(scene)
foreach(i-> series_alpha[i][] = 0.01, 1:100) # hide the first 100 series

function histogram_evolution(E, alg, f, ic_alg, n; params=PhysicalParameters())
    l = f(E, alg=alg, ic_alg=ic_alg, params=params)
    for nᵢ in n
        l2 = f(E, alg=alg_type(T=nᵢ*alg.T), ic_alg=ic_alg, params=params)
    end
    hist = fit(StatsBase.Histogram, l1, nbins=nbins, closed=:left)

    Δλ = abs.(l1-l2)./l1
    colors = [mean(Δλ[idxs_in_bin(b, hist, l1)]) for b in axes(hist.weights, 1)]
    replace!(colors, NaN=>0)

    hist_sc = plot_hist(hist, colors=colors)
end
