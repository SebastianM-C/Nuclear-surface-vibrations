#!/usr/bin/env julia
using ProgressMeter
using SymPy, JLD
using ArgParse

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
    "--energy", "-e"
        help = "The energy of the system"
        arg_type = Float64
        nargs = '+'
        default = [50.]
    "--energy_steps", "-n"
        help = "The number of energy steps"
        arg_type = Int64
        default = 14
    "--q_steps", "-m"
        help = "The number of q₀ and q₂ steps"
        arg_type = Int64
        default = 8
    end
    parsed_args = parse_args(ARGS, arg_settings)
    energy_list = parsed_args["energy"]
    n = parsed_args["energy_steps"]
    m = parsed_args["q_steps"]

    energy_list, n, m
end

function kineticEnergySlices(energy, n)
    if energy > 50 && n > 3
        Trange = collect(Iterators.flatten((linspace(0, energy - 20, n - 3),
            [energy - 10, energy - 1, energy - 0.1])))
    elseif energy > 10 && n > 3
        Trange = collect(Iterators.flatten((linspace(0, energy - 10, n - 3),
            [energy - 5, energy - 1, energy - 0.2])))
    else
        Trange = linspace(0, energy * (1 - 1e-3), n)
    end
end

function generateQ₀(energy, m, q₂)

    function findBound(; idx=1, sgn=1)
        bound = q₀roots[idx][1]
        for i in 0:5
            ϵ = sgn * 10^(-float(i))
            bound += ϵ
            while is_real(q₂[idx](energy, bound)) == true
                bound += ϵ
            end
            bound -= ϵ
        end
        if sgn == 1   # upper bound
            return linspace(N(q₀roots[idx][1] + 50 * eps()), N(bound), m)
        else          # lower bound
            if energy > 10 && idx == 2
                return collect(Iterators.flatten((.-logspace(log10(N(-bound-1)), log10(N(-bound)), m),
                                linspace(N(bound + 1.1), N(q₀roots[idx][1] - 50 * eps()), m))))
            elseif idx == 1
                if abs(N(q₀roots[idx][1]) - N(bound)) < 100 * eps()
                    return N(q₀roots[idx][1]) - 50 * eps():N(q₀roots[idx][1]) - 50 * eps()
                else
                    return linspace(N(bound), N(q₀roots[idx][1] - 50 * eps()), m / 2)
                end
            else
                return linspace(N(bound), N(q₀roots[idx][1] - 50 * eps()), m)
            end
        end
    end

    q₀ = symbols("q₀", real=true)

    q₀roots = Array{Vector{SymPy.Sym}}(2)
    q₀roots[1] = solve(q₂[1](energy, q₀))
    q₀roots[2] = solve(q₂[2](energy, q₀))

    # Find the possible values for q₀ using the root(s) of q₂
    q₀range = Array{Any}(2)
    q₂range = Array{Any}(2)

    for i in 1:2
        if length(q₀roots[i]) > 0
            if length(q₀roots[i]) == 1
                if is_real(q₂[i](energy, q₀roots[i][1]-1e3*eps())) == true      # root is the upper bound
                    q₀range[i] = findBound(idx=i, sgn=-1)   # find lower bound
                elseif is_real(q₂[i](energy, q₀roots[i][1]+1e3*eps())) == true  # root is the lower bound
                    q₀range[i] = findBound(idx=i, sgn=1)   # find upper bound
                end
            elseif is_real(q₂[i](energy, (q₀roots[i][1] + q₀roots[i][2]) / 2)) && length(q₀roots[i]) == 2 # inside
                if energy > 10
                    q₀range[i] = collect(Iterators.flatten((.-logspace(log10(N(-q₀roots[i][1]-50*eps()-1)),
                            log10(N(-q₀roots[i][1]-50*eps())), m), linspace(N(q₀roots[i][1]+50*eps()+1.1),
                            N(q₀roots[i][2]-50*eps()), 2m))))
                else
                    q₀range[i] = linspace(N(q₀roots[i][1]+50*eps()), N(q₀roots[i][2]-50*eps()), m)
                end
            else
                println("$(length(q₀roots[i])) real roots and no real values between the (first) 2 roots")
            end
          q₂range[i] = vcat(q₂[i].(energy, q₀range[i]), -q₂[i].(energy, q₀range[i]))
          q₀range[i] = Iterators.flatten((q₀range[i], q₀range[i]))
          # plot!(plt, q₀range[i], q₂[i].(energy, q₀range[i]), legend=false)
        end
    end
    q₀valid = [isassigned(q₀range, 1), isassigned(q₀range, 2)]
    q₂valid = [isassigned(q₂range, 1), isassigned(q₂range, 2)]

    Float64.(collect(Iterators.flatten(q₀range[q₀valid]))), vcat(q₂range[q₂valid]...)
end

function explore(A, B, D, energy, n, m)
    q₂ = Array{Function}(2)
    q₂[1] = (E, q₀) -> √(-(A / D) - (3 * √2 * B * q₀) / D - q₀^2 - √(A^2 + 4 * D * E +
      6 * √2 * A * B * q₀ + 18 * B^2 * q₀^2 + 8 * √2 * B * D * q₀^3) / D)
    q₂[2] = (E, q₀) -> √(-(A / D) - (3 * √2 * B * q₀) / D - q₀^2 + √(A^2 + 4 * D * E +
      6 * √2 * A * B * q₀ + 18 * B^2 * q₀^2 + 8 * √2 * B * D * q₀^3) / D)

    @inline pUpperLimit(E, p) = √(2 / A * E - p^2)

    q₀range = Array{Vector{Float64}}(n)
    q₂range = Array{Vector{Float64}}(n)
    p₀range = Array{Vector{Float64}}(n)
    p₂range = Array{Vector{Float64}}(n)

    Trange = kineticEnergySlices(energy, n)

    prog_m = Progress(n)
    # @progress for i in 1:n
    for i in 1:n
        # @show i
        q₀range[i], q₂range[i] = generateQ₀(energy - Trange[i], m, q₂)
        p₀range[i] = linspace(0, √(2/A * Trange[i]), size(q₀range[i], 1))
        p₂range[i] = real(pUpperLimit.(Trange[i]+0im, p₀range[i]))
        ProgressMeter.next!(prog_m; showvalues = [(:E, energy)])
    end

    q0list = hcat(vcat(q₀range...), vcat(q₂range...))
    p0list = hcat(vcat(p₀range...), vcat(p₂range...))

    prefix = "../output/B$B D$D E$energy"
    if !isdir(prefix)
        mkpath(prefix)
    end
    save("$prefix/z0.jld", "q0list", q0list, "p0list", p0list)
end

function main()
    # Hamiltonian parameters
    A, B, D = readdlm("param.dat")
    energy_list, n, m = input_param()    # exploration parameters

    for energy in energy_list
        explore(A, B, D, energy, n, m)
    end
end

main()
