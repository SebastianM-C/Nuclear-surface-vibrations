using PlotRecipes, RecipesBase
using StatsBase, QuadGK

using Requires

@userplot StackedHist
@userplot FitHistogram

struct StackedHist
    hists

@require StatPlots begin
    @recipe function f(h::StackedHist)
        x, y, ws, hists = h.args
        n = length(y)
        nbins = length(x) - 1
        bin_size = (last(x) - first(x)) / nbins
        if ws == nothing
            ws = @. weights(fill(1 / (n * length(y) * bin_size), length(y)))
        end

        if hists == nothing
            hists = [fit(Histogram, i, w, x, closed=:left)
                for (i, w) in zip(y, ws)]
        end

        bars = Matrix{Float64}(nbins, n)
        for (i, hist) in enumerate(hists)
            bars[:, i] = hist.weights
        end

        bar_position := :stack
        bar_width := bin_size
        ylims --> (0., 1.)
        edges = linspace(first(x) + bin_size / 2, last(x) - bin_size / 2, nbins)
        StatPlots.GroupedBar((edges, bars))
    end
end

@recipe function f(fh::FitHistogram)
    x, y, model = fh.args
    nbins = length(x) - 1
    n = length(y)
    bin_size = (last(x) - first(x)) / nbins
    ws = @. weights(fill(1 / (n * length(y) * bin_size), length(y)))
    hists = [fit(Histogram, i, w, x, closed=:left)
        for (i, w) in zip(y, ws)]

    x_data = linspace(first(x) + bin_size / 2, last(x) - bin_size / 2, nbins)
    y_data = sum(h.weights for h in hists)
    p0 = rand(1,)
    c_fit = curve_fit(model, x_data, y_data, ones(nbins), p0;
        lower=[0.], upper=[1.])

    @series begin
        label --> ""
        StackedHist((x, y, ws, hists))
    end

    α = c_fit.param[1]
    f(x) = model(x, α)

    @series begin
        seriestype := :path
        label --> L"$\alpha = "*@sprintf("%.2f", α)*L"$"
        x, f
    end

    model_hist = [1 / bin_size * quadgk(f, x[i], x[i+1])[1]
                 for i in 1:nbins]

    ylims --> (0., 1.)

    @series begin
        seriestype := :bar
        color := :transparent
        bar_width := bin_size
        label := ""
        x, model_hist
    end
end
