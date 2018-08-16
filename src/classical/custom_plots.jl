include("poincare.jl")
include("lyapunov.jl")
include("../utils.jl")
include("gali.jl")

using RecipesBase
using DiffEqBase
using Plots
using Query, StatPlots
using StatsBase
using ProgressMeter
using IntervalArithmetic
using Utils
# using PmapProgressMeter
using LaTeXStrings
# using Juno
gr()

function energy_error(sim, E, params)
    energy_err(t,u1,u2,u3,u4) = (t, H([u1,u2],[u3,u4], params) - E)
    energy_err(sol) = size(sol.u,1) > 0 ?
        [abs.(H([sol[1,j], sol[2,j]], [sol[3,j], sol[4,j]], params) - E)
        for j=1:size(sol,2)] : 0
    info("The maximum energy error during time evolution was "*
        "$(maximum(map(i->maximum(energy_err(i)), sim.u)))")

    plt = plot(ylabel="Energy error", legend=false)
    plot!(plt, sim, vars=(energy_err, 0,1,2,3,4), msc=nothing, ms=2)
end

function average(x, y)
    int = 0
    for i = 1:length(y) - 1
        int += (y[i+1] + y[i]) * (x[i+1] - x[i]) / 2
    end
    int / (x[end] - x[1])
end

function ch_max(v)
    sorted = sort(v)
    Δ = length(v) * diff(sorted) / (sorted[end] - sorted[1])
    sorted[2:end][Δ .< 10][end]
end

function hist_avg(v, t=3)
    hist = fit(Histogram, v, nbins=50, closed=:right)
    threshold = hist.edges[1][t]
    mean(v[v.>threshold])
end

function hist_max(v, t=3)
    hist = fit(Histogram, v, nbins=50, closed=:right)
    threshold = hist.edges[1][t]
    v = v[v.>threshold]
    idx = indmax(hist.weights[t+1:end])
    a = hist.edges[1][t+1:end][idx]
    b = hist.edges[1][t+1:end][idx+1]
    maximum(v[(v.>a) .& (v.<=b)])
end

function λlist(Elist, Blist=0.55, Dlist=0.4; T=12000., Ttr=5000., recompute=false)
    for D in Dlist
        @showprogress "B" for i = 1:length(Blist)
            @showprogress "λs" for j = 1:length(Elist)
                prefix = "../../output/classical/B$(Blist[i])-D$D/E$(Elist[j])"
                λs = λmap(Elist[j], B=Blist[i], D=D, T=T, Ttr=Ttr,
                    recompute=recompute)
                plt = histogram(λs, nbins=50, xlabel="\\lambda", ylabel="N",
                    label="T = $T")
                savefig(plt, "$prefix/lyapunov_hist.pdf")
                if !any(ismatch.(r"poincare_lyapunov.*pdf", readdir(prefix))) || recompute
                    coloredpoincare(Elist[j], λs, name="lyapunov", B=Blist[i], D=D)
                end
            end
        end
    end
end


"""
    mean_as_function_of_B(value::Symbol, B, Einterval::Interval=0..Inf;
           reduction=ch_max, plt=plot(), ylabel="", fnt=font(12, "Times"),
           width=800, height=600, filename="val(E)")

Compute the mean of a stored `value` as a funciton of `B` considering the
given energy interval `Einterval`. Return the corresponding `DataFrame`
"""
function mean_as_function_of_B(value::Symbol, B, Einterval::Interval=0..Inf;
        reduction=ch_max, plt=plot(), ylabel="", fnt=font(12, "Times"),
        width=800, height=600, filename="val(E)")
    df = concat(r"z0.csv", location="classical/B$(B[1])-D0.4",
        re=r"E[0-9]+\.[0-9]+", filter=[:E, value]) |> @filter(_.E ∈ Einterval) |> DataFrame
    df_v = by(df, :E, df->DataFrame(val = reduction(df[value]))) |>
        @orderby(_.E) |> DataFrame
    df_v[:B] = fill(B[1], size(df_v, 1))

    df_v |> @df plot(:E, :val, m=2, xlabel=L"E", ylabel=ylabel,
        framestyle=:box, legend=false,
        size=(width,height),
        guidefont=fnt, tickfont=fnt)
    savefig("../../output/classical/B$(B[1])-D0.4/$filename.pdf")

    for i in 2:length(B)
        df = concat(r"z0.csv", location="classical/B$(B[i])-D0.4",
            re=r"E[0-9]+\.[0-9]+", filter=[:E, value]) |> @filter(_.E ∈ Einterval) |> DataFrame
        df_ = by(df, :E, df->DataFrame(val = reduction(df[value]))) |>
            @orderby(_.E) |> DataFrame
        df_ |> @df plot(:E, :val, m=2, xlabel=L"E", ylabel=ylabel,
            framestyle=:box, legend=false,
            size=(width,height),
            guidefont=fnt, tickfont=fnt)
        savefig("../../output/classical/B$(B[i])-D0.4/$filename.pdf")
        df_[:B] = fill(B[i], size(df_, 1))
        append!(df_v, df_[names(df_v)])
    end

    by(df_v |> @filter(_.E ∈ Einterval) |> DataFrame, :B,
        df->DataFrame(v = average(df[:E], df[:val]))) |>
            @orderby(_.B) |> DataFrame
            # @df plot!(plt, :B, :v, m=2, xlabel=L"B", ylabel=ylabel,
                # label=L"$E \in "*"$Einterval\$")
end

function mean_as_function_of_B(value::Symbol, B, Eintervals::NTuple{N, Interval};
        reduction=ch_max, plt=plot(), ylabel="", fnt=font(12, "Times"),
        width=800, height=600, filename="val(E)") where N
    for i=1:N
        plt = mean_as_function_of_B(value, B, Eintervals[i], reduction=reduction,
            plt=plt, ylabel=ylabel, fnt=fnt, width=width, height=height,
            filename=filename)
    end
    plt
end

function galilist(Elist, Blist=0.55, Dlist=0.4; tmax=500, recompute=false)
    for D in Dlist
        @progress "B" for i = 1:length(Blist)
            @progress "GALI_2s" for j = 1:length(Elist)
                prefix = "../../output/classical/B$(Blist[i])-D$D/E$(Elist[j])"
                galis = galimap(Elist[j], B=Blist[i], D=D, tmax=tmax,
                    recompute=recompute)
                plt = histogram(galis, nbins=50, xlabel=L"$t_{th} GALI_2$",
                    ylabel="N", label="ratio = $(count(galis .< tmax) / length(galis))")
                savefig(plt, "$prefix/gali_hist.pdf")
                if !any(ismatch.(r"poincare_gali.*pdf", readdir(prefix))) || recompute
                    coloredpoincare(Elist[j], galis, name="gali", B=Blist[i], D=D)
                end
            end
        end
    end
end
