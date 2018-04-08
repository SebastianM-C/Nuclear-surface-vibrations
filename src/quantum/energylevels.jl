#!/usr/bin/env julia

module EnergyLevels

export levels

include("hamiltonian.jl")
using .Hamiltonian
using TimerOutputs
using JLD

"""
    sortlvl!(E, eigv)

Take the real part and sort the energy levels and the corresponding
eigenvectors.

## Arguments
- `E`: energy levels
- `eigv`: corresponding eigenvectors
"""
function sortlvl!(E, eigv)
    E = real(E)
    eigv = real(eigv)
    idx = sortperm(E)

    E[idx], eigv[:, idx]
end

"""
    levels(n::Integer, f=0.1; a=1., b=0.55, d=0.4)

Compute the first `floor(f*N)` eigenvalues, the corresponding eigenvectors
of the Hamiltonian, the maximum coefficient index for the eigenvectors
and the index corresponding to the ordering of the basis.

The basis is oredered as follows
```math
|0,0⟩, |0,1⟩, |0,2⟩ … |0,n⟩, |1,0⟩ … |1,n-1⟩ … |i,0⟩, |i,n-i⟩ … |n-1,0⟩
```

If those values were previously computed, they will be used instead.

## Arguments
- `n::Integer`: the dimension of the harmonic oscillator basis
in one of the directions.
- `f = 0.1`: the fraction of the number of eigenvalues to be computed.
For a given value, `Int(floor(f*N))` eigenvalues will be computed,
where `N = n*(n+1)/2`.

## Keyword Arguments
- `a = 1.`:   the Hamiltonian A parameter
- `b = 0.55`: the Hamiltonian B parameter
- `d = 0.4`:  the Hamiltonian D parameter

See also [`compute_hamiltonian`](@ref)
"""
function levels(n::Integer, f=0.1; a=1., b=0.55, d=0.4)
    prefix = "../../output/quantum/n$n-b$b-d$d"
    if !isdir(prefix)
        mkpath(prefix)
    end
    # Use already computed values when available
    if isfile("$prefix/eigensystem-f$f.jld")
        info("Loading previously computed values.")
        E, eigv, nconv, niter = load("$prefix/eigensystem-f$f.jld",
            "E", "eigv", "nconv", "niter")
    else
        @timeit "Hamiltonian computation for n$n b$b d$d" begin
            H = compute_hamiltonian(n, a=a, b=b, d=d)
        end
        N = n*(n+1)/2
        nev = Int(floor(f*N))
        label = "Diagonalisation for n$n f$f b$b d$d"
        @timeit label begin
            E, eigv, nconv, niter, nmult, resid = eigs(H, nev=nev, which=:SM)
        end
        save("$prefix/eigensystem-f$f.jld", "E", E, "eigv", eigv, "nconv",
            nconv, "niter", niter, "nmult", nmult, "resid", resid)
        saveperf(prefix, label)
    end
    nconv != nev && warn("Not all eigenvalues converged.")
    niter >= 300 && warn("Reached maximum number of iterations.")

    E, eigv = sortlvl!(E, eigv)

    return E, eigv
end

function parse_label(label)
    re = r"n([0-9]+) f(0.[0-9]+) b([0-9]+\.?[0-9]+) d([0-9]+\.?[0-9]+)"
    n, f, b, d = match(re, label).captures
    parse(Int, n), float.((f, b, d))...
end

function saveperf(prefix, label)
    to = TimerOutputs.DEFAULT_TIMER
    n, f, b, d  = parse_label(label)
    params = ([n], [f], [b], [d])

    if !isfile("$prefix/perf_data.jld")
        # 1 gibibyte (GiB) = 2³⁰ bytes (B)
        save("$prefix/perf_data.jld", "cores", [Int(ceil(Sys.CPU_CORES / 2))],
            "memory", [Sys.total_memory() / 2^30], "params", params,
            "t", [TimerOutputs.time(to[label]) / 1e9],  # 1s = 1e9 ns
            "allocated", [TimerOutputs.allocated(to[label]) / 2^30],
            "ncalls", [TimerOutputs.ncalls(to[label])])
    else
        cores, memory, oldparams, t, allocated, ncalls = load("$prefix/perf_data.jld",
            "cores", "memory", "params", "t", "allocated", "ncalls")
        for i=1:length(params) push!(oldparams[i], params[i][1]) end
        push!(cores, Int(ceil(Sys.CPU_CORES / 2)))
        push!(memory, Sys.total_memory() / 2^30)
        save("$prefix/perf_data.jld", "cores", cores, "memory", memory,
            "params", oldparams,
            "t", push!(t, TimerOutputs.time(to[label]) / 1e9),
            "allocated", push!(allocated,
                TimerOutputs.allocated(to[label]) / 2^30),
            "ncalls", push!(ncalls, TimerOutputs.ncalls(to[label])))
    end
end

"""
    irreducible_reps(E, ket, ε=1e-8)

Separate the energy levels according to the 3 irreducible
representations of the ``C_{3v}`` group.

## Arguments
- `E`: energy levels
- `ket`: the states in the number operator representation
- `ε`: the maximum difference between two degenerate levels
"""
function irreducible_reps(E, ket, ε=1e-8)

end

end  # module EnergyLevels
