include("poincare.jl")
include("lyapunov.jl")
include("../utils.jl")
include("gali.jl")

using RecipesBase
using DiffEqBase
using Plots
using Query, StatPlots
using ProgressMeter
using IntervalArithmetic
using Utils
using PmapProgressMeter
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

function λBlist(B, Einterval::Interval=0..Inf, plt=plot())
    function ch_max(v)
        sorted = sort(v)
        Δ = length(v) * diff(sorted) / (sorted[end] - sorted[1])
        sorted[2:end][Δ .< 10][end]
    end

    df = concat(r"z0.csv", location="classical/B$(B[1])-D0.4",
        re=r"E[0-9]+\.[0-9]+", filter=[:E, :λs])
    df_λ = by(df, :E, df->DataFrame(λ = ch_max(df[:λs]))) |>
        @orderby(_.E) |> DataFrame
    df_λ[:B] = fill(B[1], size(df_λ, 1))

    df_λ |> @df plot(:E, :λ, m=4, xlabel="E", ylabel="\\lambda", label="B = $(B[1])")
    savefig("../../output/classical/B$(B[1])-D0.4/lambda(E).pdf")

    for i in 2:length(B)
        df = concat(r"z0.csv", location="classical/B$(B[i])-D0.4",
            re=r"E[0-9]+\.[0-9]+", filter=[:E, :λs])
        df_ = by(df, :E, df->DataFrame(λ = ch_max(df[:λs]))) |>
            @orderby(_.E) |> DataFrame
        df_λ |> @df plot(:E, :λ, m=4, xlabel="E", ylabel="\\lambda", label="B = $(B[i])")
        savefig("../../output/classical/B$(B[i])-D0.4/lambda(E).pdf")
        df_[:B] = fill(B[i], size(df_, 1))
        append!(df_λ, df_[names(df_λ)])
    end

    by(df_λ |> @filter(_.E ∈ Einterval) |> DataFrame, :B,
        df->DataFrame(λ = average(df[:E], df[:λ]))) |>
            @orderby(_.B) |>
            @df plot!(plt, :B, :λ, m=4, xlabel=L"$B$", ylabel=L"\lambda",
                label=L"$E \in "*"$Einterval\$")
end

function λBlist(B, Eintervals::NTuple{N, Interval}, plt=plot()) where N
    for i=1:N
        plt = λBlist(B, Eintervals[i], plt)
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

function galiBlist(B, Einterval::Interval=0..Inf, plt=plot(); tmax=500)
    ratio(v) = count(v .< tmax) / length(v)
    df = concat(r"z0.csv", location="classical/B$(B[1])-D0.4",
        re=r"E[0-9]+\.[0-9]+", filter=[:E, :gali])
    df_g = by(df, :E, df->DataFrame(f = ratio(df[:gali]))) |>
        @orderby(_.E) |> DataFrame
    df_g[:B] = fill(B[1], size(df_g, 1))

    for i in 2:length(B)
        df = concat(r"z0.csv", location="classical/B$(B[i])-D0.4",
            re=r"E[0-9]+\.[0-9]+", filter=[:E, :gali])
        df_ = by(df, :E, df->DataFrame(f = ratio(df[:gali]))) |>
            @orderby(_.E) |> DataFrame
        df_[:B] = fill(B[i], size(df_, 1))
        append!(df_g, df_[names(df_g)])
    end

    by(df_g |> @filter(_.E ∈ Einterval) |> DataFrame, :B,
        df->DataFrame(f = average(df[:E], df[:f]))) |>
            @orderby(_.B) |>
            @df plot!(plt, :B, :f, m=4, xlabel="B", ylabel=L"$f_{chaotic}$",
                label="E in $Einterval")
end

function galiBlist(B, Eintervals::NTuple{N, Interval}, plt=plot()) where N
    for i=1:N
        plt = galiBlist(B, Eintervals[i], plt)
    end
    plt
end
