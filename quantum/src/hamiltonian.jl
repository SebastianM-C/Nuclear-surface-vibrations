#!/usr/bin/env julia
using ArgParse
using PyCall

function input_param()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "-n"
            help = "Size of the diagonalization basis"
            arg_type = Int64
            default = 120
        "-a", "-A"
            help = "Hamiltonian A parameter"
            arg_type = Float64
            default = 1.
        "-b", "-B"
            help = "Hamiltonian B parameter"
            arg_type = Float64
            default = 0.55
        "-d", "-D"
            help = "Hamiltonian D parameter"
            arg_type = Float64
            default = 0.4

    end
    parsed_args = parse_args(ARGS, arg_settings)
    n = parsed_args["n"]
    a = parsed_args["a"]
    b = parsed_args["b"]
    d = parsed_args["d"]

    return n, a, b, d
end

#    Compute the matrix element on harmonic oscillator states
#             + k    l
#    < m | (a )  (a)  | n >

@inline function elem(m::Integer, n::Integer, k::Integer, l::Integer)
    (m != n - l + k || n < l) && return 0.

    elem = 1.
    # Apply anihilation operator
    for i = 0:(l - 1)
        elem *= √(n - i)
    end
    # Apply creation operator
    for i = 1:k
        elem *= √(n - l + i)
    end

    return elem
end

function generate_hamiltonian(n, a, b, d)
    N = Int(n * (n + 1) / 2)
    H = Matrix{Float64}(N, N)
    idx = Matrix{Int64}(2, N)
    # Compute index
    k = 0
    for i = 0:(n - 1), j = 0:(n - i - 1)
        k += 1
        idx[1, k] = i
        idx[2, k] = j
    end
    # Compute Hamiltonian
    for i = 1:N, j = 1:N
        m1 = idx[1, j]
        m2 = idx[2, j]
        n1 = idx[1, i]
        n2 = idx[2, i]

        H[j, i] = a * (elem(m1, n1, 1, 1) * elem(m2, n2, 0, 0) +
                     elem(m1, n1, 0, 0) * elem(m2, n2, 1, 1))
        H[j, i] += 0.25 * b * (3 *
                     (elem(m1, n1, 1, 0) * elem(m2, n2, 2, 0))
               + 3 * elem(m1, n1, 0, 1) * elem(m2, n2, 0, 2)
                   - elem(m1, n1, 3, 0) * elem(m2, n2, 0, 0)
                   - elem(m1, n1, 0, 3) * elem(m2, n2, 0, 0))
        H[j, i] += 0.75 * b * ((
                     elem(m1, n1, 0, 1) * elem(m2, n2, 2, 0)
                   + elem(m1, n1, 1, 0) * elem(m2, n2, 0, 2)
                   - elem(m1, n1, 1, 2) * elem(m2, n2, 0, 0))
                   - elem(m1, n1, 2, 1) * elem(m2, n2, 0, 0)
               + 2 * elem(m1, n1, 0, 1) * elem(m2, n2, 1, 1)
               + 2 * elem(m1, n1, 1, 0) * elem(m2, n2, 1, 1))
        H[j, i] += 0.375 * d * (
                     elem(m1, n1, 2, 2) * elem(m2, n2, 0, 0)
                   + elem(m1, n1, 0, 0) * elem(m2, n2, 2, 2))
        H[j, i] += 0.125 * d * (
                     elem(m1, n1, 2, 0) * elem(m2, n2, 0, 2)   +
                   + elem(m1, n1, 0, 2) * elem(m2, n2, 2, 0))  +
         + 0.5 * d * elem(m1, n1, 1, 1) * elem(m2, n2, 1, 1)
        H[j, i] += 0.250 * d * ((
                     elem(m1, n1, 1, 3) * elem(m2, n2, 0, 0)
                   + elem(m1, n1, 3, 1) * elem(m2, n2, 0, 0)
                   + elem(m1, n1, 0, 0) * elem(m2, n2, 1, 3))
                   + (elem(m1, n1, 0, 0) * elem(m2, n2, 3, 1)
                   + elem(m1, n1, 0, 2) * elem(m2, n2, 1, 1)
                   + elem(m1, n1, 2, 0) * elem(m2, n2, 1, 1))
                   + elem(m1, n1, 1, 1) * elem(m2, n2, 0, 2)
                   + elem(m1, n1, 1, 1) * elem(m2, n2, 2, 0))
        H[j, i] += 0.0625 * d * ((
                     elem(m1, n1, 4, 0) * elem(m2, n2, 0, 0)
                   + elem(m1, n1, 0, 4) * elem(m2, n2, 0, 0)
                   + elem(m1, n1, 0, 0) * elem(m2, n2, 4, 0))
                   + elem(m1, n1, 0, 0) * elem(m2, n2, 0, 4)
               + 2 * elem(m1, n1, 2, 0) * elem(m2, n2, 2, 0)
               + 2 * elem(m1, n1, 0, 2) * elem(m2, n2, 0, 2))
   end

   return H
end

function main()
    n, a, b, d = input_param()
    prefix = "../output/B$b D$d N$n"
    if !isdir(prefix)
        mkpath(prefix)
    end
    @time H = generate_hamiltonian(n, a, b, d)
    @pyimport numpy as np
    np.savez_compressed("$prefix/hamilt.npz", H=H)
end

main()
