module Recipes

export StackedHist, FitHistogram, fit_histogram, fithistogram

using PlotRecipes, RecipesBase, LaTeXStrings
using StatsBase, QuadGK, LsqFit
using Requires

@userplot StackedHist
@userplot FitHistogram

@require StatPlots begin
    @recipe function f(h::StackedHist)
        x, y, hists = h.args
        n = length(y)
        nbins = length(x) - 1
        bin_size = (last(x) - first(x)) / nbins

        if hists == nothing
            ws = @. weights(fill(1 / (n * length(y) * bin_size), length(y)))
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

function fit_histogram(x, hists, model, nbins, lu)
    y = sum(h.weights for h in hists)
    p0 = rand(1,)
    c_fit = curve_fit(model, x, y, ones(nbins), p0;
        lower=lu[1], upper=lu[2])
end

function fit_histogram(x, y, model, lu=([0.], [1.]))
    n = length(y)
    nbins = length(x) - 1
    bin_size = (last(x) - first(x)) / nbins

    ws = @. weights(fill(1 / (n * length(y) * bin_size), length(y)))
    hists = [fit(Histogram, i, w, x, closed=:left)
        for (i, w) in zip(y, ws)]
    fit_histogram(linspace(first(x) + bin_size / 2, last(x) - bin_size / 2, nbins),
        hists, model, nbins, lu)
end

@recipe function f(fh::FitHistogram)
    x, y, models, lus, pnames = fh.args
    n = length(y)
    nbins = length(x) - 1
    bin_size = (last(x) - first(x)) / nbins

    ws = @. weights(fill(1 / (n * length(y) * bin_size), length(y)))
    hists = [fit(Histogram, i, w, x, closed=:left)
        for (i, w) in zip(y, ws)]

    @series begin
        StackedHist((x, y, hists))
    end

    x_data = linspace(first(x) + bin_size / 2, last(x) - bin_size / 2, nbins)
    if typeof(models) <: AbstractArray
        ps = [fit_histogram(x_data, hists, model, nbins, lu).param
            for (model, lu) in zip(models, lus)]
        fs = [x->model(x, p) for (model, p) in zip(models, ps)]
        lab = [L"$"*name*" = "@sprintf("%.2f", length(p)>1 ? p : p[1])*L"$"
            for (p, name) in zip(ps, pnames)]
    else
        p = fit_histogram(x_data, hists, models, nbins, lus).param
        fs = x->models(x, p)
        lab = L"$"*pnames*" = "*@sprintf("%.2f", length(p)>1 ? p : p[1])*L"$"
        model_hist = [1 / bin_size * quadgk(fs, x[i], x[i+1])[1]
                     for i in 1:nbins]
    end

    ylims --> (0., 1.)

    @series begin
        seriestype := :path
        label --> lab
        x_data, fs
    end

    if !(typeof(models) <: AbstractArray)
        @series begin
            seriestype := :bar
            color := :transparent
            bar_width := bin_size
            label := ""
            x, model_hist
        end
    end

end

end  # module Recipes
