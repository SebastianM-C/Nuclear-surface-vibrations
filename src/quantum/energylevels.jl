#!/usr/bin/env julia

module EnergyLevels

export diagonalize, elvls, nlvls, add_max_δ, irreducible_reps

include("hamiltonian.jl")
using .Hamiltonian
using TimerOutputs
using JLD
using DataFrames, Query, CSV

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
        @info("Loading previously computed values for n $n b $b d $d.")
        E, eigv, nconv, niter = load("$prefix/eigensystem-f$f.jld",
            "E", "eigv", "nconv", "niter")
    else
        @timeit "Hamiltonian computation for n$n b$b d$d" begin
            H = compute_hamiltonian(n, a=a, b=b, d=d)
        end
        gc()
        label = "Diagonalisation for n$n f$f b$b d$d"
        @timeit label begin
            E, eigv, nconv, niter, nmult, resid = eigs(H, nev=nev, which=:SM)
        end
        save("$prefix/eigensystem-f$f.jld", "E", E, "eigv", eigv, "nconv",
            nconv, "niter", niter, "nmult", nmult, "resid", resid)
        saveperf(prefix, label)
    end
    nconv != nev && @warn("Not all eigenvalues converged.")
    niter >= 300 && @warn("Reached maximum number of iterations.")

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
- `n::Integer`: the dimension of the harmonic oscillator basis in one of
the directions.
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
        nconv != nev && @warn("Not all eigenvalues converged.")
        niter >= 300 && @warn("Reached maximum number of iterations.")
        return sort(real(E))
    else
        return diagonalize(n, f, a=a, b=b, d=d)[1]
    end
end

function resize(r_E, E)
    E1, E2 = length(r_E) > length(E) ? (r_E[1:length(E)], E) : (r_E, E[1:length(r_E)])
end

"""
    δ(r_E, E)

Compute the maximum energy difference between two sets of energy levels.
"""
function δ(r_E, E)
    E1, E2 = resize(r_E, E)
    maximum(abs.(E1 - E2))
end

"""
    δ(df::AbstractDataFrame)

Compute the maximum energy difference for a `DataFrame` taking as reference
the the row with the greatest diagonalization basis `n`. The given `DataFrame`
should have a single value for Hamiltonian parameters (given by the
`b` and `d` columns).
"""
function δ(df::AbstractDataFrame)
    allsame(v) = isempty(v) || all(isequal(first(v)), v)
    allsame(df[:b]) && allsame(df[:d]) || @error("The parameters are not the same.")
    ref = df[indmax(df[:n]), :]
    r_E = elvls(ref[:n][1], ref[:f][1], b=ref[:b][1], d=ref[:d][1])
    [δ(r_E, elvls(df[i,:][:n][1], df[i,:][:f][1], b=df[i,:][:b][1],
        d=df[i,:][:d][1])) for i=1:size(df, 1)]
end

"""
    add_max_δ(df::AbstractDataFrame)

Add a column corresponding to the maximum energy difference for a `DataFrame`
taking as reference the the row with the greatest diagonalization basis `n`.
See also [`δ`](@ref).
"""
function add_max_δ(df::AbstractDataFrame)
    td = by(df, [:b,:d]) do df
        DataFrame(t=df[:t], max_δ=δ(df))
    end |> @map({_.t,_.max_δ}) |> DataFrame
    data = join(df, td, on=:t, kind=:inner) |>
        @map({_.cores,_.n,_.f,levels=nlvls.(_.n,_.f),_.b,t=_.t/3600,_.max_δ}) |>
        DataFrame
end

"""
    filter_symmetric(E, eigv, n; ϵ=1e-6)

Filter the levels corresponding to the symmetric representation of
the ``C_{3v}`` group by considering the reflection with respect to the
``y`` axis. If ``R_y`` is the unitary operator corresponding to the
reflection, then the eigenvectors ``|\\Psi\\rangle_s`` belonging to the symmetric
representation are invariant to the reflections mentioned earlier, that
is
```math
R_y |\\Psi\\rangle_s = |\\Psi\\rangle_s.
```

The function returns a `BitArray` for the levels belonging to the
symmetric representations and the maximum difference on each column
for each eigenvector ``R_y |\\Psi\\rangle - |\\Psi\\rangle``.

## Arguments
- `eigv`: the corresponding eigenvectors
- `n`: the dimension of the harmonic oscillator basis in one of the directions
- `ϵ = 1e-6`: the maximum error for the separation of the symmetric representation
"""
function filter_symmetric(eigv, n; ϵ=1e-6)
    N = Int(n*(n+1)/2)
    idx = Hamiltonian.index(n)

    U = Diagonal([(-1)^(idx[2, i]) for i=1:N])

    Δ = abs.(U * eigv - eigv)
    symm = BitArray(all(Δ[:,i] .< ϵ) for i=1:size(Δ, 2))
    return symm, [maximum(Δ[:,i]) for i=1:size(Δ, 2)]
end

"""
    filter_bidimensional(E; ε=1e-9)

Filter the energy levels corresponding to the bidimensional representation
of the ``C_{3v}``. The levels are filtered using the fact that they are doublby
degenerate. Thus we select the levels for which
```math
E_{n+1} - E_n < \\varepsilon
```
In case of 2 consecutive differences, both below ``\\varepsilon``, we choose
the smaller one. The function returns a `BitArray` for the unique levels
belonging to the bidimensional representations and a `BitArray` with all
the selected ones.

## Arguments
- `E`: energy levels
- `ε = 1e-9`: the maximum difference between two degenerate levels
"""
function filter_bidimensional(E; ε=1e-9)
    ΔE = diff(E) .< ε
    bd = falses(length(E))

    k = 0
    for i=1:length(E)-1
        k = ΔE[i] ? k+1 : 0
        if k == 2
            diff(E)[i] > diff(E)[i-1] ? ΔE[i] = false : ΔE[i-1] = false
        end
        bd[i] = bd[i] || ΔE[i]
        bd[i+1] = ΔE[i]
    end
    return ΔE, bd
end

function verify_reps(ΔE, symm, bd)
    longest_seq = 0
    k = 0
    idx = 0
    for i=1:length(ΔE)
        k = ΔE[i] ? k+1 : 0
        k > longest_seq && (longest_seq = k; idx = i)
    end
    longest_seq != 1 &&
        @warn("Found a sequence of $longest_seq equal differences at $idx")
    !all(.!(symm .& bd)) && @warn("The intersection of the symmetric "*
        "representation and the bidimensional one has $(count(symm .& bd)) "*
        "elements")
end

"""
    irreducible_reps(E, eigv, n, ϵ=1e-6, ε=1e-9)

Separate the energy levels according to the 3 irreducible
representations of the ``C_{3v}`` group. See also [`filter_symmetric`](@ref)
and [`filter_bidimensional`](@ref).

## Arguments
- `E`: energy levels
- `eigv`: the corresponding eigenvectors
- `n`: the dimension of the harmonic oscillator basis in one of the directions
- `ϵ = 1e-6`: the maximum error for the separation of the symmetric representation
- `ε = 1e-9`: the maximum difference between two degenerate levels
"""
function irreducible_reps(E, eigv, n; ϵ=1e-6, ε=1e-9)
    ΔE, bd = filter_bidimensional(E, ε=ε)
    symm, Δ = filter_symmetric(eigv, n, ϵ=ϵ)

    verify_reps(ΔE, symm, bd)

    E[1:length(E)-1][ΔE], E[symm], E[.!(symm .| bd)]
end

end  # module EnergyLevels
