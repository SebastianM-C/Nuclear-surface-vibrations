#!/usr/bin/env julia
using JLD
using ArgParse
using Plots, LaTeXStrings
using ProgressMeter
# plotlyjs()
pyplot()
# pgfplots()

include("chaos_limit.jl")
using ChaosLimit

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
    "--energy", "-e"
        help = "The energy of the system"
        arg_type = Float64
        nargs = '+'
        default = [100.]
    end
    parsed_args = parse_args(ARGS, arg_settings)
    E_list = parsed_args["energy"]
end

function plot_λs(λs, d0, dt, tr, tmax, prefix)
    n = size(λs, 1)
    ch_lim, n_chaotic = chaos_limit(λs)
    chaotic = λs[λs .> ch_lim]
    regular = λs[λs .<= ch_lim]
    α = n_chaotic / n
    plt1 = plot(xaxis="index", yaxis=L"\lambda", legend=false,
        title=L"$D(0)=$"*" $d0, "*L"$\Delta t=$"*" $dt, tr = $tr");
    for r in regular
        scatter!(plt1, [Float64.(findin(λs, r))], [r], markerstrokecolor=nothing);
    end
    for c in sort(chaotic)
        scatter!(plt1, [Float64.(findin(λs, c))], [c], markerstrokecolor=nothing,
            palette=:grays);
    end
    # scatter!(plt1, findin(λs, chaotic), chaotic, color=:black);
    plot!(plt1, x->ch_lim, label="chaotic threshold")
    if α != 0
        plt2 = histogram(chaotic, xaxis=L"\lambda", yaxis=L"$N$",
            label="chaotic:$n_chaotic\nratio:$α");
    else
        plt2 = histogram(λs, xaxis=L"\lambda", yaxis=L"$N$",
            label="chaotic:$n_chaotic\nratio:$α");
    end
    plt = plot(plt1, plt2, size=(900, 700));
    savefig(plt, "$prefix/color_lyapunov_$tmax.pdf")
end

function main()
    # Hamiltonian parameters
    A, B, D = readdlm("param.dat")
    E_list = input_param()
    prog_m = Progress(size(E_list, 1), "Energy list: ")
    for E in E_list
        prefix = "../output/B$B D$D E$E"
        if !isdir(prefix)
            mkpath(prefix)
        end
        if isfile("$prefix/z0.jld")
            q0list, p0list = load("$prefix/z0.jld", "q0list", "p0list")
        else
            error("$prefix/z0.jld not found! Generate the initial conditions.")
        end
        if isfile("$prefix/lyapunov.jld")
            λs, d0, dt, tr, tmax, n =
                load("$prefix/lyapunov.jld", "λs", "d0", "dt", "tr", "tmax", "n")
        else
            error("$prefix/lyapunov.jld not found! Generate Lyapunov coefficients.")
        end
        println("Ploting the previous results with:")
        println("d0: $d0\ndt: $dt\ntr: $tr\ntmax: $tmax\nn: $n")
        plot_λs(λs, d0, dt, tr, tmax, prefix)
        ProgressMeter.next!(prog_m; showvalues = [(:E, E)])
    end
end

main()
