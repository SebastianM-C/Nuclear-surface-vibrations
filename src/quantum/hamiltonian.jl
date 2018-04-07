#!/usr/bin/env julia

__precompile__()

"""
This module exports a function that generates the Hamiltonian as a
dense matrix in the basis given by the eigenstates of an isotropic
double harmonic oscillator.
```math
  H = A \\left( a_1^\\dagger a_1 + a_2^\\dagger a_2 \\right)
  + \\frac{B}{4} \\bigg[ \\left( 3 a_1^\\dagger {a_2^\\dagger}^2 + 3 a_1 a_2^2
                             - {a_1^\\dagger}^3 - a_1^3 \\right)
   + 3 \\left( a_1 {a_2^\\dagger}^2 + a_1^\\dagger a_2^2 - a_1^\\dagger a_1^2 - {a_1^\\dagger}^2 a_1
           + 2 a_1 a_2^\\dagger a_2 + 2 a_1^\\dagger a_2^\\dagger a_2
        \\right) \\bigg]
   + \\frac{D}{16} \\bigg[ 6 \\left( {a_1^\\dagger}^2 a_1^2 + {a_2^\\dagger}^2 a_2^2 \\right)
                      + 2 \\left( a_1^2 {a_2^\\dagger}^2 + {a_1^\\dagger}^2 a_2^2 \\right)
                      + 8 a_1^\\dagger a_1 a_2^\\dagger a_2
   + 4 \\left(a_1^\\dagger a_1^3 + {a_1^\\dagger}^3 a_1 + a_2^\\dagger a_2^3 + {a_2^\\dagger}^3 a_2
   + a_1^2 a_2^\\dagger a_2 + {a_1^\\dagger}^2 a_2^\\dagger a_2 + a_1^\\dagger a_1 a_2^2 + a_1^\\dagger a_1 {a_2^\\dagger}^2
      \\right)
   + \\left( {a_1^\\dagger}^4 + a_1^4 + {a_2^\\dagger}^4 + a_2^4
   + 2 {a_1^\\dagger}^2 {a_2^\\dagger}^2 + 2 a_1^2 a_2^2
    \\right)
                      \\bigg].
```
"""
module Hamiltonian

export compute_hamiltonian

"""
    elem(m::Integer, n::Integer, k::Integer, l::Integer)

Compute the matrix element on harmonic oscillator states
``
\\langle m | (a^{\\dagger})^k a^l | n \\rangle
``
"""
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

"""
    compute_dense_hamiltonian(n::Integer; a=1., b=0.55, d=0.4)

Compute the Hamiltonian with the given parameters as a dense matrix.

## Arguments
- `n::Integer`: the dimension of the harmonic oscillator basis
in one of the directions. The dimension of the basis will be
``N = \\frac{1}{2}n(n+1)`` and the matrix will have ``N^2`` elements.

## Keyword arguments
- `a = 1.`:   the Hamiltonian A parameter
- `b = 0.55`: the Hamiltonian B parameter
- `d = 0.4`:  the Hamiltonian D parameter

## Examples

```jldoctest
julia> compute_dense_hamiltonian(3)
6×6 Array{Float64,2}:
 0.0  0.0    0.0        0.0       0.0     0.0
 0.0  1.0    0.0        0.0       0.825   0.0
 0.0  0.0    2.3        0.583363  0.0     0.1
 0.0  0.0    0.583363   1.0       0.0    -0.583363
 0.0  0.825  0.0        0.0       2.2     0.0
 0.0  0.0    0.1       -0.583363  0.0     2.3

julia> compute_dense_hamiltonian(3, b=0., d=0.)
6×6 Array{Float64,2}:
6×6 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  2.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  2.0  0.0
 0.0  0.0  0.0  0.0  0.0  2.0

```
"""
function compute_dense_hamiltonian(n::Integer; a=1., b=0.55, d=0.4)
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

"""
    compute_hamiltonian(n::Integer; a=1., b=0.55, d=0.4)

Compute the Hamiltonian with the given parameters as a sparse matrix. See
also [`compute_dense_hamiltonian`](@ref).

## Arguments
- `n::Integer`: the dimension of the harmonic oscillator basis
in one of the directions. The dimension of the basis will be
``N = \\frac{1}{2}n(n+1)`` and the matrix will have ``N^2`` elements.

## Keyword arguments
- `a = 1.`:   the Hamiltonian A parameter
- `b = 0.55`: the Hamiltonian B parameter
- `d = 0.4`:  the Hamiltonian D parameter

## Examples

```jldoctest
julia> compute_hamiltonian(3)
6×6 SparseMatrixCSC{Float64,Int64} with 13 stored entries:
  [2, 2]  =  1.0
  [5, 2]  =  0.825
  [3, 3]  =  2.3
  [4, 3]  =  0.583363
  [6, 3]  =  0.1
  [3, 4]  =  0.583363
  [4, 4]  =  1.0
  [6, 4]  =  -0.583363
  [2, 5]  =  0.825
  [5, 5]  =  2.2
  [3, 6]  =  0.1
  [4, 6]  =  -0.583363
  [6, 6]  =  2.3

 ```
"""
function compute_hamiltonian(n::Integer; a=1., b=0.55, d=0.4)
    sparse(compute_dense_hamiltonian(n, a=a, b=b, d=d))
end

end  # module Hamiltonian
