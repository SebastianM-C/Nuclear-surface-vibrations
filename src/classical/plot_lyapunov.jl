#!/usr/bin/env julia
addprocs(Int(Sys.CPU_CORES / 2));
# addprocs(6);
using JLD
using ArgParse
using Plots, LaTeXStrings
using ProgressMeter
using PmapProgressMeter
# plotlyjs()
# pyplot()
pgfplots()

include("chaos_limit.jl")
using ChaosLimit

@everywhere begin
include("problem.jl")
include("lyapunov.jl")
end
@everywhere begin
    using HamiltonEqs, Lyapunov
    using OrdinaryDiffEq, DiffEqCallbacks
    using ParallelDataTransfer

    @inline V(q₀, q₂) = A / 2 * (q₀^2 + q₂^2) + B / √2 * q₀ * (3 * q₂^2 - q₀^2) + D / 4 * (q₀^2 + q₂^2)^2;
    @inline T(p₀, p₂) = A / 2 * (p₀^2 + p₂^2);
    @inline H(q₀, q₂, p₀, p₂) = T(p₀, p₂) + V(q₀, q₂);

    function g(u, resid)
      resid[1] = H(u...) - E
      resid[2] = 0
      resid[3] = 0
      resid[4] = 0
    end
end

# Get the energy list and the parameters for the computation of the
# maximal Lyapunov exponent
function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
    "--energy", "-e"
        help = "The energy of the system"
        arg_type = Float64
        nargs = '+'
        default = [120.]
    "--tmax", "-t"
        help = "Simulation time"
        arg_type = Float64
        default = 5e4
    "--d0"
        help = "Initial separation"
        arg_type = Float64
        default = 5e-11
    "--dt"
        help = "Rescaling time interval"
        arg_type = Float64
        default = 1.5
    "--solver"
        help = "Integration method"
        arg_type = String
        default = "DPRKN12"
    "--cache"
        help = "Plot the previously computed values for λs"
        action = :store_true
    end
    parsed_args = parse_args(ARGS, arg_settings)
    E_list = parsed_args["energy"]
    tmax = parsed_args["tmax"]
    d0 = parsed_args["d0"]
    dt = parsed_args["dt"]
    solver = parsed_args["solver"]
    cache = parsed_args["cache"]
    # E = 50.
    # tmax = 5e4
    # d0 = 5e-11
    # dt = 1.5
    tr = 10^4 * d0    # maximum distance trashold in which the
                      # linearised dynamics approximation is valid

    if solver == "DPRKN12"
        kwargs = Dict(:solver=>DPRKN12(), :abstol=>1e-14, :reltol=>1e-14)
        defProb = defineProblem
    elseif solver == "Vern9+ManifoldProjection"
        cb = ManifoldProjection(g)
        kwargs = Dict(:solver=>Vern9(), :abstol=>1e-14, :reltol=>1e-14, :callback=>cb)
        defProb = defineProblem2
    else
        @error("No configuration available for $solver")
    end
    E_list, tmax, d0, dt, tr, solver, cache, kwargs, defProb
end

function plot_λs(λs, d0, dt, tr, tmax, prefix)
    n = size(λs, 1)
    ch_lim, n_chaotic = chaos_limit(λs)
    α = n_chaotic / n
    plt1 = scatter(λs, xaxis="index", yaxis=L"\lambda", label=nothing,
        title=L"$D(0)=$"*" $d0, "*L"$\Delta t=$"*" $dt, tr = $tr");
    plot!(plt1, x->ch_lim, label="chaotic threshold")
    if α != 0
        chaotic = λs[λs .> ch_lim]
        plt2 = histogram(chaotic, xaxis=L"\lambda", yaxis=L"$N$",
            label="chaotic:$n_chaotic\nratio:$α");
    else
        plt2 = histogram(λs, xaxis=L"\lambda", yaxis=L"$N$",
            label="chaotic:$n_chaotic\nratio:$α");
    end
    plt = plot(plt1, plt2, size=(900, 700));
    savefig(plt, "$prefix/lyapunov_$tmax.pdf")
end

function compute_λs(q0list, p0list, tmax, d0, dt, tr, kwargs, prefix)
    n = size(q0list, 1)    # number of initial conditions
    λs = SharedArray{Float64}(n)
    pmap(i->(prob = defProb(q0list[i,:], p0list[i,:], (0., tmax));
            λs[i] = compute_lyapunov(prob, d0=d0, dt=dt, threshold=tr,
                                    diff_eq_kwargs=kwargs)),
        Progress(n),
        1:n)

    save("$prefix/lyapunov.jld", "λs", λs, "d0", d0, "dt", dt, "tr", tr,
        "tmax", tmax, "n", n);
    return λs
end

function main()
    # Hamiltonian parameters
    A, B, D = readdlm("param.dat")
    E_list, tmax, d0, dt, tr, solver, cache, kwargs, defProb = input_param()
    # Broadcast parameters to all workers
    sendto(workers(), A=A, B=B, D=D, defProb=defProb)

    prog_m = Progress(size(E_list, 1), "Energy list: ")
    for E in E_list
        sendto(workers(), E=E)
        prefix = "../output/B$B D$D E$E"
        if !isdir(prefix)
            mkpath(prefix)
        end
        if isfile("$prefix/z0.jld")
            q0list, p0list = load("$prefix/z0.jld", "q0list", "p0list")
        else
            @error("$prefix/z0.jld not found! Generate the initial conditions.")
        end
        if !cache
            λs = compute_λs(q0list, p0list, tmax, d0, dt, tr, kwargs, prefix)
            plot_λs(λs, d0, dt, tr, tmax, prefix)
        else
            if isfile("$prefix/lyapunov.jld")
                λs, d0, dt, tr, tmax, n =
                    load("$prefix/lyapunov.jld", "λs", "d0", "dt", "tr", "tmax", "n")
            else
                @error("$prefix/lyapunov.jld not found! Generate Lyapunov coefficients.")
            end
            println("Ploting the previous results with:")
            println("d0: $d0\ndt: $dt\ntr: $tr\ntmax: $tmax\nn: $n")
            plot_λs(λs, d0, dt, tr, tmax, prefix)
        end
        ProgressMeter.next!(prog_m; showvalues = [(:E, E)])
    end
end

main()
