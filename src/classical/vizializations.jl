using Makie
using StatsBase
using StatMakie
using Statistics
using IntervalArithmetic

function plot_sim(sim, sc=Scene(resolution=(1000,1000)); ms=0.005,
        colors=axes(sim, 1))
    points = Point2f0[]
    all_colors = Float32[]
    m = minimum(colors)
    M = maximum(colors)
    Δ = (M-m) / length(colors)

    for i in axes(sim, 1)
        simᵢ = sim[i]
        if length(sim[i]) ≠ 0
            append!(all_colors, Iterators.repeated(colors[i], length(simᵢ)))
            append!(points, (Point2f0(simᵢ[i,1], simᵢ[i,2]) for i in axes(simᵢ, 1)))
        end
    end
    colormap = to_colormap(:viridis, length(colors))
    pushfirst!(colormap, RGBAf0(0.1, 0.1, 0.1, 0.1))
    scatter!(sc, points, markersize=ms, color=all_colors, colormap=colormap,
        colorrange=(m - 2Δ, M))
end

function plot_hist(hist; colors=Float32.(axes(hist.weights, 1)))
    m = minimum(colors)
    M = maximum(colors)
    Δ = (M-m) / length(colors)
    colormap = to_colormap(:viridis, length(colors))
    pushfirst!(colormap, RGBAf0(0.1, 0.1, 0.1, 0.1))
    plot(hist, color=colors, colormap=colormap,
        colorrange=(m - 2Δ, M))
end

function setup_click(scene)
    idx_node = Node(0)
    foreach(scene.events.mousebuttons) do buttons
        if ispressed(scene, Mouse.left) && AbstractPlotting.is_mouseinside(scene)
            plt, click_idx = Makie.mouse_selection(scene)
            return push!(idx_node, click_idx)
        end
        return nothing
    end
    return idx_node
end

function select_series(v, scene, idx, hist_sc, l, hist)
    original_sc_colors = copy(scene[end][:color][])
    original_hist_colors = copy(hist_sc[end][:color][])
    zero_sc = minimum(scene[end][:colorrange][])
    zero_hist = minimum(hist_sc[end][:colorrange][])

    foreach(idx) do click_idx
        println("scatter ", click_idx)
        len = 1
        series_idx = 0
        splot = scene[end]
        if click_idx ≠ 0
            for i in axes(v, 1)
                if len ≤ click_idx < len+size(v[i], 1)
                    series_idx = i
                    println("in series $i")
                else
                    splot[:color][][len:len+size(v[i], 1)-1] .= zero_sc
                    splot[:color][] = splot[:color][]
                end
                len += size(v[i], 1)
            end
            bin_idx = bin_with_val(l[series_idx], hist)
            hist_sc[end][:color][][setdiff(axes(hist_sc[end][:color][], 1), bin_idx)] .= zero_hist
            hist_sc[end][:color][] = hist_sc[end][:color][]
        else
            println("outside")
            splot[:color][] = copy(original_sc_colors)
            hist_sc[end][:color][] = copy(original_hist_colors)
        end
        return nothing
    end
end

bin_with_val(val, hist) = searchsortedfirst(hist.edges[1], val) - 1

function idxs_in_bin(i, hist, val)
    h = hist.edges[1]
    bin = interval(h[i], h[i + 1])
    idx = findall(x -> x ∈ bin, val)
    # println(idx)
    return idx
end

function select_bin(scene, idx, l, hist, scatter_sc, v)
    original_sc_colors = copy(scatter_sc[end][:color][])
    original_hist_colors = copy(scene[end][:color][])
    zero_sc = minimum(scatter_sc[end][:colorrange][])
    zero_hist = minimum(scene[end][:colorrange][])

    foreach(idx) do click_idx
        println("histogram ", click_idx)
        if click_idx ≠ 0
            l_idx = idxs_in_bin(click_idx, hist, l)
            len = 1
            splot = scatter_sc[end]
            for i in axes(v, 1)
                if i ∉ l_idx
                    splot[:color][][len:len+size(v[i], 1)-1] .= zero_sc
                    splot[:color][] = splot[:color][]
                end
                len += size(v[i], 1)
            end
            scene[end][:color][][setdiff(axes(scene[end][:color][], 1), click_idx)] .= zero_hist
            scene[end][:color][] = scene[end][:color][]
        else
            println("outside")
            scatter_sc[end][:color][] = copy(original_sc_colors)
            scene[end][:color][] = copy(original_hist_colors)
        end
        return nothing
    end
end

function poincare_explorer(E, alg, f, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, ms=0.05, rootkw=(xrtol=1e-6, atol=1e-6), n=10)
    q0, p0 = initial_conditions(E, alg=ic_alg)
    alg_type = typeof(alg)
    sim = poincaremap(q0, p0, params=params, sgn=sgn, axis=axis, t=alg.T, rootkw=rootkw)
    l1 = f(E, alg=alg, ic_alg=ic_alg, params=params)
    l2 = f(E, alg=alg_type(T=n*alg.T), ic_alg=ic_alg, params=params)
    # l1 = f(q0, p0, alg, params=params)
    # l2 = f(q0, p0, alg_type(T=n*alg.T), params=params)
    hist = fit(StatsBase.Histogram, l1, nbins=nbins, closed=:left)

    Δλ = abs.(l1-l2)./l1
    colors = [mean(Δλ[idxs_in_bin(b, hist, l1)]) for b in axes(hist.weights, 1)]
    replace!(colors, NaN=>0)

    hist_sc = plot_hist(hist, colors=colors)
    scatter_sc = plot_sim(sim, ms=ms, colors=l1)
    if axis == 3
        scatter_sc[Axis][:names, :axisnames] = ("q₂","p₂")
    else
        scatter_sc[Axis][:names, :axisnames] = ("q₁","p₁")
    end
    sc = AbstractPlotting.vbox(scatter_sc, hist_sc)

    scatter_idx = setup_click(scatter_sc)
    hist_idx = setup_click(hist_sc)

    select_series(sim, scatter_sc, scatter_idx, hist_sc, l1, hist)
    select_bin(hist_sc, hist_idx, l1, hist, scatter_sc, sim)

    return sc
end

function poincare_explorer(E, alg::LyapunovAlgorithm, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, ms=0.05, rootkw=(xrtol=1e-6, atol=1e-6))
    scene = poincare_explorer(E, alg, λmap, ic_alg; params=params, axis=axis, sgn=sgn,
        nbins=nbins, ms=ms, rootkw=rootkw)
    scene.children[2][Axis][:names, :axisnames] = ("λ","N")

    return scene
end

function poincare_explorer(E, alg::DInftyAlgorithm, ic_alg; params=PhysicalParameters(),
        axis=3, sgn=-1, nbins=50, ms=0.05, rootkw=(xrtol=1e-6, atol=1e-6))
    poincare_explorer(E, alg, d∞, ic_alg; params=params, axis=axis, sgn=sgn,
        nbins=nbins, ms=ms, rootkw=rootkw)
end

# colorlegend(s1.children[2], s1.children[2][end][:colormap][], s1.children[2][end][:colorrange])
E = 1000.
ic_alg = PoincareRand(n=200)
ic_alg = PoincareUniform(n=9,m=9)
alg = DynSys()

s1 = poincare_explorer(E, alg, ic_alg)

s2 = poincare_explorer(E, DInftyAlgorithm(), ic_alg)

center!(s2.children[1])[Axis][:names, :axisnames] = ("q₂","p₂")
s1.children[2][end][:color][]

q0, p0 = initial_conditions(E, alg=ic_alg)

l1 = λmap(q0, p0, DynSys())
l2 = λmap(q0, p0, DynSys(T=1e5))
l3 = λmap(E, alg=DynSys(T=2e5), ic_alg=PoincareRand(n=200))

hist = fit(StatsBase.Histogram, l1, nbins=50, closed=:left)
Δλ = abs.(l1-l2)./l1
colors = [mean(Δλ[idxs_in_bin(b, hist, l1)]) for b in axes(hist.weights, 1)]
replace!(colors, NaN=>0)

m = minimum(colors)
M = maximum(colors)
Δ = (M-m) / length(colors)
colormap = to_colormap(:viridis, length(colors))
pushfirst!(colormap, RGBAf0(0.1, 0.1, 0.1, 0.1))
p=plot(hist, color=colors, colormap=colormap,
    colorrange=(m - 2Δ, M))
Plots.scatter(q0[:,2], p0[:,2], mz=Δλ)


p1=plot(fit(StatsBase.Histogram, l1, nbins=50, closed=:left))
p2=plot(fit(StatsBase.Histogram, l2, nbins=50))

AbstractPlotting.vbox(p,p1,p2)

h = fit(StatsBase.Histogram, l1, nbins=50)
colors[idxs_in_bin(4, h, l1)]|>mean

plot_hist(fit(StatsBase.Histogram, l1, nbins=50), colors=colors)

with_logger(dbg) do
    λmap(E, alg=DynSys(T=2e5), ic_alg=PoincareRand(n=3), params=PhysicalParameters(B=0.2))
end

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

ui_width = 260
n = [1,4,10]
scene = Scene(resolution = (1000, 1000))
scatter!(rand(10))


ui = Scene(scene, lift(x-> IRect(0, 0, ui_width, widths(x)[2]), pixelarea(scene)))
plot_scene = Scene(scene, lift(x-> IRect(ui_width, 0, widths(x) .- Vec(ui_width, 0)), pixelarea(scene)))
theme(ui)[:plot] = NT(raw = true)
campixel!(ui)
translate!(ui, 10, 50, 0)

n_slider = AbstractPlotting.textslider(ui, n, "n")

ui
AbstractPlotting.hbox!(ui.plots)

foreach(n_slider) do n
    @show n
    return nothing
end

import Plots
Plots.plot(
    Plots.histogram(l1, nbins=50, xlims=(-0.01,0.23)),
    Plots.histogram(l2, nbins=50, xlims=(-0.01,0.23)),
    Plots.histogram(l3, nbins=50, xlims=(-0.01,0.23)),
    layout=(1,3), size=(1400,500)
)

using Reactive
