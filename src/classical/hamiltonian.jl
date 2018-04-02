#!/usr/bin/env julia
module Hamiltonian

export H, T, V, q̇, ṗ

T(p) = A / 2 * norm(p)^2
# V(q) = 1/2 * (q[1]^2 + q[2]^2 + 2q[1]^2 * q[2]- 2/3 * q[2]^3)
V(q) = A / 2 * (q[1]^2 + q[2]^2) + B / √2 * q[1] * (3 * q[2]^2 - q[1]^2) + D / 4 * (q[1]^2 + q[2]^2)^2;
const A, B, D = isfile("parameters") ? readdlm("parameters") : 1., 0.55, 0.4
H(q, p) = T(p) + V(q)

function q̇(t, q, p, dq)
  p₀, p₂ = p
  dq[1] = A * p₀
  dq[2] = A * p₂
end

function ṗ(t, q, p, dp)
  q₀, q₂ = q
  dp[1] = -A * q₀ - 3 * B / √2 * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2)
  dp[2] = -q₂ * (A + 3 * √2 * B * q₀ + D * (q₀^2 + q₂^2))
end

end  # module Hamiltonian
