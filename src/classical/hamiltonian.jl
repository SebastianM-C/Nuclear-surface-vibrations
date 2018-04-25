#!/usr/bin/env julia
module Hamiltonian

export H, T, V, q̇, ṗ

function T(p, params)
  A = params
  A / 2 * norm(p)^2
end
# V(q) = 1/2 * (q[1]^2 + q[2]^2 + 2q[1]^2 * q[2]- 2/3 * q[2]^3)
function V(q, params)
  A, B, D = params
  A / 2 * (q[1]^2 + q[2]^2) + B / √2 * q[1] * (3 * q[2]^2 - q[1]^2) + D / 4 * (q[1]^2 + q[2]^2)^2;
end

H(p, q, params=(1, 0.55, 0.4)) = T(p, params=params[1]) + V(q, params=params[2:3])

function q̇(dq, p, q, params=1., t)
  A = params
  p₀, p₂ = p
  dq[1] = A * p₀
  dq[2] = A * p₂
end

function ṗ(dq, p, q, params=(1, 0.55, 0.4), t)
  A, B, D = params
  q₀, q₂ = q
  dp[1] = -A * q₀ - 3 * B / √2 * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2)
  dp[2] = -q₂ * (A + 3 * √2 * B * q₀ + D * (q₀^2 + q₂^2))
end

end  # module Hamiltonian
