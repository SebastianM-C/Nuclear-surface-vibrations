#!/usr/bin/env julia
using JLD
using Plots, LaTeXStrings
using ArgParse
using ProgressMeter
pgfplots()

include("problem.jl")
include("poincare.jl")
include("chaos_limit.jl")
using HamiltonEqs, Poincare, ChaosLimit

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "--energy", "-e"
            help = "The energy of the system"
            arg_type = Float64
            nargs = '+'
            default = [50.]
         "--tmax", "-t"
            help = "Simulation time"
            arg_type = Float64
            default = 1e3
         "--axis", "-a"
            help = "Axis along to choose the intersection plane"
            arg_type = Int64
            default = 1
         "--sign", "-s"
            help = "Sign of the momentum at the intersection plane"
            arg_type = Int64
            default = 1
         "--solver"
            help = "Integration method"
            arg_type = String
            default = "Vern9+ManifoldProjection"
         "--type"
            help = "Poincare section plot type. Choose from chaotic_only,
                    regular_only, chaotic_black, heatmap, default, all"
            arg_type = String
            default = "default"
    end
    parsed_args = parse_args(ARGS, arg_settings)
    E_list = parsed_args["energy"]
    tmax = parsed_args["tmax"]
    axis = parsed_args["axis"]
    sgn = parsed_args["sign"]
    solver = parsed_args["solver"]
    plot_type = parsed_args["type"]
    tspan = (0., tmax)

    E_list, tspan, axis, sgn, solver, plot_type
end

function main()
    # Hamiltonian parameters
    A, B, D = readdlm("param.dat")
    E_list, tspan, axis, sgn, solver, plot_type = input_param()
    # E = 50.

    if solver == "Vern9+ManifoldProjection"
        defProb = defineProblem2
    elseif solver == "KahanLi8"
        defProb = defineProblem
    end
    # prob = defProb([0.,0.], [0.,0.], tspan)

    @inline V(q₀, q₂) = A / 2 * (q₀^2 + q₂^2) + B / √2 * q₀ * (3 * q₂^2 - q₀^2) + D / 4 * (q₀^2 + q₂^2)^2;
    @inline T(p₀, p₂) = A / 2 * (p₀^2 + p₂^2);
    @inline H(q₀, q₂, p₀, p₂) = T(p₀, p₂) + V(q₀, q₂);

    function poincarePlot(E, plot_type)
        prefix = "../output/B$B D$D E$E"
        if !isdir(prefix)
            mkpath(prefix)
        end
        if isfile("$prefix/z0.jld")
            q0list, p0list = load("$prefix/z0.jld", "q0list", "p0list")
        else
            error("$prefix/z0.jld not found! Generate the initial conditions.")
        end

        if plot_type == "default"
            plt = createMap(defProb, q0list, p0list, tspan, solver, axis, sgn, H, E)
        else
            if isfile("$prefix/lyapunov.jld")
                λs = load("$prefix/lyapunov.jld", "λs")
            else
                error("$prefix/lyapunov.jld not found! Generate Lyapunov coefficients.")
            end
            ch_lim, n_chaotic = chaos_limit(λs)
            chaotic = findin(λs, λs[λs .> ch_lim])
            regular = findin(λs, λs[λs .<= ch_lim])
            if plot_type == "heatmap"
                plt = createMap(defProb, q0list, p0list, tspan, λs, solver, axis, sgn, H, E)
            elseif plot_type == "chaotic_only"
                if size(chaotic, 1) == 0
                    warn("No chaotic trajectories.")
                    return nothing
                end
                plt = createMap(defProb, q0list[chaotic,:], p0list[chaotic,:],
                    tspan, λs[chaotic,:], solver, axis, sgn, H, E)
            elseif plot_type == "regular_only"
                if size(regular, 1) == 0
                    warn("No regular trajectories.")
                    return nothing
                end
                plt = createMap(defProb, q0list[regular,:], p0list[regular,:],
                    tspan, λs[regular,:], solver, axis, sgn, H, E)
            elseif plot_type == "chaotic_black"
                if size(regular, 1) == 0
                    warn("No regular trajectories.")
                    return nothing
                end
                plt = createMap(defProb, q0list[regular,:], p0list[regular,:],
                    tspan, solver, axis, sgn, H, E, progress_info="Regular: ")
                if size(chaotic, 1) == 0
                    warn("No chaotic trajectories.")
                    return nothing
                end
                # sort_idx = sortperm(λs[λs .> ch_lim])
                # sorted_ch = chaotic[sort_idx]
                # createMap!(plt, defProb, q0list[sorted_ch,:], p0list[sorted_ch,:],
                #     tspan, solver, axis, sgn, H, E, progress_info="Chaotic: ",
                #     plot_kwargs=Dict(:markersize=>1, :palette=>:grays))
                createMap!(plt, defProb, q0list[chaotic,:], p0list[chaotic,:],
                    tspan, solver, axis, sgn, H, E, progress_info="Chaotic: ",
                    plot_kwargs=Dict(:markersize=>1, :color=>:black))
            else
                error("No configuration for $plot_type")
            end
        end
        # savefig(plt, "$prefix/poincare_$plot_type-ax$axis-t_$(tspan[2])_sgn$sgn.pdf");
        savefig(plt, "$prefix/poincare_$plot_type-ax$axis-t_$(tspan[2])_sgn$sgn.tex");
    end

    prog_m = Progress(size(E_list, 1), "Energy list: ")
    for E in E_list
        if plot_type == "all"
            for t in ("default", "heatmap", "chaotic_only", "regular_only", "chaotic_black")
                poincarePlot(E, t)
            end
        else
            poincarePlot(E, plot_type)
        end
        ProgressMeter.next!(prog_m; showvalues = [(:E, E)])
    end
end

main()
