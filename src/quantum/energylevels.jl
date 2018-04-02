#!/usr/bin/env julia

module EnergyLevels

include("hamiltonian.jl")
using Hamiltonian
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
    H = sparse(generate_hamiltonian(n, a=a, b=b, d=d))
    N = n*(n+1)/2
    nev = Int(floor(f*N))
    # Use already computed values when available
    # if isfile("eigensystem.jld")
    #     info("Loading previously computed values.")
    #     E, eigv, nconv, niter, nmult, resid = load("eigensystem.jld",
    #         "E", "eigv", "nconv", "niter", "nmult", "resid", "f")
    # else
        @timeit "Diagonalisation" begin
            E, eigv, nconv, niter, nmult, resid =
                eigs(H, nev=nev, which=:SM)
        end
    #     save("eigensystem.jld", "E", E, "eigv", eigv, "nconv", nconv,
    #         "niter", niter, "nmult", nmult, "resid", resid, "f", f)
    # end
    nconv != f*N && warn("Not all eigenvalues converged.")
    niter >= 300 && warn("Reached maximum number of iterations.")

    E, eigv = sortlvl!(E, eigv)
    max_c_idx = [indmax(abs.(eigv[:,i])) for i=1:nev]
    index = [(n1, n2) for n1=0:n-1 for n2=0:n-1-n1]

    print_timer()

    return E, eigv, max_c_idx, index
end

"""
    irreducible_reps(E, ket, ε=1e-8)

Separate the energy levels according to the 3 irreducible
representations of the ``C_{3v}`` group.

## Arguments
- `E`: energy levels
- `ket`: the states in the number operator representation
- `ε`: the
"""
function irreducible_reps(E, ket, ε=1e-8)


end
using Plots
Int(260*261/2)
E, eigv, max_c_idx, index = levels(260, 0.7)
plot(eigv2[1:size(eigv,1),908])
plot(E2[1:length(E)] - E)
maximum(abs.(E2[1:length(E)] - E))
E2, eigv2, max_c_idx2, index2 = levels(140, 0.7)
index[max_c_idx][103]
E[98:104]

H=sparse(generate_hamiltonian(120))
heatmap(generate_hamiltonian(65), aspect_ratio=1, clims=(-1,1))

end  # module EnergyLevels
