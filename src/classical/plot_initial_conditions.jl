#!/usr/bin/env julia
using JLD
using Plots#, LaTeXStrings
using ArgParse
using ProgressMeter
plotly()

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
    "--energy", "-e"
         help = "The energy of the system"
         arg_type = Float64
         nargs = '+'
         default = [120.]
    end
    parsed_args = parse_args(ARGS, arg_settings)
    E_list = parsed_args["energy"]
    # E = 50.

    E_list
end

function plot_potential(V, q0list, E, prefix)
    plot(xaxis="q0", yaxis="q2", zaxis="V");
    q₀range = linspace(minimum(q0list[:,1]) * 1.2, maximum(q0list[:,1]) * 1.2, 100)
    q₂range = linspace(minimum(q0list[:,2]) * 1.2, maximum(q0list[:,2]) * 1.2, 100)
    surface!(q₀range, q₂range, V, zlims=(0, 1.5E), clims=(0, 1.5E));
    surface!(q₀range, q₂range, (x,y)->E, α=0.6);
    scatter3d!(q0list[:,1], q0list[:,2], V.(q0list[:,1], q0list[:,2]), ms=2, legend=false);
    savefig("$prefix/3D_plot_V.html");
end

function contour_plot_potential(V, q0list, E, prefix)
    plot(xaxis="q0", yaxis="q2");
    q₀range = linspace(minimum(q0list[:,1]) * 1.2, maximum(q0list[:,1]) * 1.2, 100)
    q₂range = linspace(minimum(q0list[:,2]) * 1.2, maximum(q0list[:,2]) * 1.2, 100)
    contour!(q₀range, q₂range, V, zlims=(0, 1.5E), clims=(0, 1.5E));
    scatter!(q0list[:,1], q0list[:,2], ms=2, legend=false);
    savefig("$prefix/contour_plot_V.html");
end

function main()
    # Hamiltonian parameters
    A, B, D = readdlm("param.dat")
    E_list = input_param()

    @inline V(q₀, q₂) = A / 2 * (q₀^2 + q₂^2) + B / √2 * q₀ * (3 * q₂^2 - q₀^2) + D / 4 * (q₀^2 + q₂^2)^2;
    @inline T(p₀, p₂) = A / 2 * (p₀^2 + p₂^2);
    @inline H(q₀, q₂, p₀, p₂) = T(p₀, p₂) + V(q₀, q₂);

    prog_m = Progress(size(E_list, 1), "Energy list: ")
    for E in E_list
        prefix = "../output/B$B D$D E$E"
        if !isdir(prefix)
            mkpath(prefix)
        end
        q0list, p0list = load("$prefix/z0.jld", "q0list", "p0list")

        plot_potential(V, q0list, E, prefix)
        contour_plot_potential(V, q0list, E, prefix)
        ProgressMeter.next!(prog_m; showvalues = [(:E, E)])
    end
end

main()
