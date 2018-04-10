#!/usr/bin/env julia

module EnergyLevels

export diagonalize, elvls, nlvls, δ

include("hamiltonian.jl")
using .Hamiltonian
using TimerOutputs
using JLD
using DataFrames, CSV

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

nlvls(n, f) = Int(floor(n*(n+1)/2 * f))

"""
    diagonalize(n::Integer, f=0.075; a=1., b=0.55, d=0.4)

Compute the first `floor(f*N)` eigenvalues, the corresponding eigenvectors
of the Hamiltonian.

The basis is oredered as follows
```math
|0,0⟩, |0,1⟩, |0,2⟩ … |0,n⟩, |1,0⟩ … |1,n-1⟩ … |i,0⟩, |i,n-i⟩ … |n-1,0⟩
```

If those values were previously computed, they will be used instead.

## Arguments
- `n::Integer`: the dimension of the harmonic oscillator basis
in one of the directions.
- `f = 0.075`: the fraction of the number of eigenvalues to be computed.
For a given value, `Int(floor(f*N))` eigenvalues will be computed,
where `N = n*(n+1)/2`.

## Keyword Arguments
- `a = 1.`:   the Hamiltonian A parameter
- `b = 0.55`: the Hamiltonian B parameter
- `d = 0.4`:  the Hamiltonian D parameter

See also [`compute_hamiltonian`](@ref)
"""
function diagonalize(n::Integer, f=0.075; a=1., b=0.55, d=0.4)
    prefix = "../../output/quantum/n$n-b$b-d$d"
    nev = nlvls(n, f)
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

    if !isfile("$prefix/perf_data.csv")
        # 1 gibibyte (GiB) = 2³⁰ bytes (B)
        df = DataFrame()
        df[:cores] = [Int(ceil(Sys.CPU_CORES / 2))]
        df[:memory] = [Sys.total_memory() / 2^30]
        df[:n] = [n]
        df[:f] = [f]
        df[:b] = [b]
        df[:d] = [d]
        df[:t] = [TimerOutputs.time(to[label]) / 1e9]  # 1s = 1e9 ns
        df[:allocated] = [TimerOutputs.allocated(to[label]) / 2^30]
        df[:ncalls] = [TimerOutputs.ncalls(to[label])]
    else
        df = CSV.read("$prefix/perf_data.csv")
        push!(df, [Int(ceil(Sys.CPU_CORES / 2)), Sys.total_memory() / 2^30,
            n, f, b, d, TimerOutputs.time(to[label]) / 1e9,
            TimerOutputs.allocated(to[label]) / 2^30,
            TimerOutputs.ncalls(to[label])])
    end
    CSV.write("$prefix/perf_data.csv", df)
end

"""
    elvls(n::Integer, f=0.075; a=1., b=0.55, d=0.4)

Return the energy levels corresponding to the given parameters from previous
cpmputations. If there is no previous computation, the compute them.
See also [`diagonalize`](@ref).

## Arguments
- `n::Integer`: the dimension of the harmonic oscillator basis
in one of the directions.
- `f = 0.075`: the fraction of the number of eigenvalues to be computed.
For a given value, `Int(floor(f*N))` eigenvalues will be computed,
where `N = n*(n+1)/2`.

## Keyword Arguments
- `a = 1.`:   the Hamiltonian A parameter
- `b = 0.55`: the Hamiltonian B parameter
- `d = 0.4`:  the Hamiltonian D parameter
"""
function elvls(n::Integer, f=0.075; a=1., b=0.55, d=0.4)
    prefix = "../../output/quantum/n$n-b$b-d$d"
    nev = nlvls(n, f)
    if !isdir(prefix)
        mkpath(prefix)
    end
    # Use already computed values when available
    if isfile("$prefix/eigensystem-f$f.jld")
        E, nconv, niter = load("$prefix/eigensystem-f$f.jld",
            "E", "nconv", "niter")
        nconv != nev && warn("Not all eigenvalues converged.")
        niter >= 300 && warn("Reached maximum number of iterations.")
        return sort(real(E))
    else
        return diagonalize(n, f, a=a, b=b, d=d)[1]
    end
end

"""
    δ(r_E, E)

Compute the maximum energy difference between two sets of energy levels.
"""
function δ(r_E, E)
    E1, E2 = length(r_E) > length(E) ? (r_E[1:length(E)], E) : (r_E, E[1:length(r_E)])
    maximum(abs.(E1 - E2))
end

"""
    δ(df::AbstractDataFrame)

Compute the maximum energy difference for a `DataFrame` taking as reference
the the row with the greatest number of levels. The given `DataFrame`
should have a single value for Hamiltonian parameters (given by the
`b` and `d` columns).
"""
function δ(df::AbstractDataFrame)
    allsame(v) = isempty(v) || all(isequal(first(v)), v)
    allsame(df[:b]) && allsame(df[:d]) || error("The parameters are not the same.")
    ref = df[indmax(nlvls.(df[:n], df[:f])), :]
    r_E = elvls(ref[:n][1], ref[:f][1], b=ref[:b][1], d=ref[:d][1])
    [δ(r_E, elvls(df[i,:][:n][1], df[i,:][:f][1], b=df[i,:][:b][1],
        d=df[i,:][:d][1])) for i=1:size(df, 1)]
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
