#!/usr/bin/env julia
module Hamiltonian

export H, T, V, ż, ṗ, q̇

using StaticArrays
using Parameters
using LinearAlgebra

function T(p, A)
  A / 2 * norm(p)^2
end
# V(q) = 1/2 * (q[1]^2 + q[2]^2 + 2q[1]^2 * q[2]- 2/3 * q[2]^3)
function V(q, params)
  @unpack A, B, D = params
  A / 2 * (q[1]^2 + q[2]^2) + B / √2 * q[1] * (3 * q[2]^2 - q[1]^2) + D / 4 * (q[1]^2 + q[2]^2)^2
end

function Vjac!(J, q, params)
  @unpack A, B, D = params
  J[1,1] = A * q[1] + B / √2 * (3 * q[2]^2 - q[1]^2) -  B * √2 * q[1]^2 + D * q[1] * (q[1]^2 + q[2]^2)
  J[1,2] = A * q[2] + 3*√2 * q[1]* q[2] + D * q[2] * (q[1]^2 + q[2]^2)
end

H(p, q, params=(A=1, B=0.55, D=0.4)) = T(p, params.A) + V(q, params)

@inbounds @inline function ż(z, p, t)
  A, B, D = p
  p₀, p₂ = z[1:2]
  q₀, q₂ = z[3:4]

  return SVector{4}(
    -A * q₀ - 3 * B / √2 * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2),
    -q₂ * (A + 3 * √2 * B * q₀ + D * (q₀^2 + q₂^2)),
    A * p₀,
    A * p₂
  )
end

@inbounds @inline function ṗ(p, q, params, t)
    A, B, D = params
    dp1 = -A * q[1] - 3 * B / √2 * (q[2]^2 - q[1]^2) - D * q[1] * (q[1]^2 + q[2]^2)
    dp2 = -q[2] * (A + 3 * √2 * B * q[1] + D * (q[1]^2 + q[2]^2))
    return SVector{2}(dp1, dp2)
end

@inbounds @inline function q̇(p, q, params, t)
    params[1] * p
end

function q̇(dq, p, q, params, t)
  A = params
  p₀, p₂ = p
  dq[1] = A * p₀
  dq[2] = A * p₂
end

function ṗ(dq, p, q, params, t)
  A, B, D = params
  q₀, q₂ = q
  dp[1] = -A * q₀ - 3 * B / √2 * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2)
  dp[2] = -q₂ * (A + 3 * √2 * B * q₀ + D * (q₀^2 + q₂^2))
end

end  # module Hamiltonian
