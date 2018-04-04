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

See also [`generate_hamiltonian`](@ref)
"""
function levels(n::Integer, f=0.1; a=1., b=0.55, d=0.4)
    @timeit "Hamiltonian generation for B$b N$n"
        H = sparse(generate_hamiltonian(n, a=a, b=b, d=d))
    end
    N = n*(n+1)/2
    nev = Int(floor(f*N))
    prefix = "../../output/quantum/N$n-B$b-D$d"
    if !isdir(prefix)
        mkpath(prefix)
    end
    # Use already computed values when available
    if isfile("$prefix/eigensystem-f$f.jld")
        info("Loading previously computed values.")
        E, eigv, nconv, niter, nmult, resid = load("$prefix/eigensystem.jld",
            "E", "eigv", "nconv", "niter", "nmult", "resid")
    else
        label = "Diagonalisation for B$b D$d N$n f$f"
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

function saveperf(prefix, label)
    to = TimerOutputs.DEFAULT_TIMER

    if !isfile("$prefix/raw_perf_data.jld")
        save("$prefix/raw_perf_data.jld", "label", [label],
            "t", [TimerOutputs.time(to[label]) / 1e9],
            # 1 gibibyte (GiB) = 1073741824 bytes (B)
            "allocated", [TimerOutputs.allocated(to[label]) / 1073741824],
            "ncalls", [TimerOutputs.ncalls(to[label])])
    else
        oldlabel, t, allocated, ncalls = load("$prefix/raw_perf_data.jld",
            "label", "t", "allocated", "ncalls")
        save("$prefix/raw_perf_data.jld", "label", push!(oldlabel, label),
            "t", push!(t, TimerOutputs.time(to[label]) / 1e9),
            # 1 gibibyte (GiB) = 1073741824 bytes (B)
            "allocated", push!(allocated,
                TimerOutputs.allocated(to[label]) / 1073741824),
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
