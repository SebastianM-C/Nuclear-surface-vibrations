#!/usr/bin/env julia
module HamiltonEqs
using OrdinaryDiffEq, ParameterizedFunctions

export defineProblem, defineProblem2

function defineProblem(q0, p0, tspan)
  # Parameters
  A, B, D = readdlm("param.dat")
  function HamiltonEqs_q(t, q, p, dq)
    p₀, p₂ = p
    dq[1] = A * p₀
    dq[2] = A * p₂
  end

  function HamiltonEqs_p(t, q, p, dp)
    q₀, q₂ = q
    dp[1] = -A * q₀ - 3 * B / √2 * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2)
    dp[2] = -q₂ * (A + 3 * √2 * B * q₀ + D * (q₀^2 + q₂^2))
  end

  prob = DynamicalODEProblem{true}(HamiltonEqs_q, HamiltonEqs_p, q0, p0, tspan)
end

h_eqs = @ode_def_noinvjac HamiltEqs begin
  dq₀ = A * p₀
  dq₂ = A * p₂
  dp₀ = -A * q₀ - 3 * B / √2 * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2)
  dp₂ = -q₂ * (A + 3 * √2 * B * q₀ + D * (q₀^2 + q₂^2))
end A=>1 B=>0.55 D=>0.4

function defineProblem2(q0, p0, tspan)
  # Parameters
  A, B, D = readdlm("param.dat")
  h_eqs.A = A
  h_eqs.B = B
  h_eqs.D = D
  u0 = hcat(q0, p0)

  prob2 = ODEProblem(h_eqs, u0, tspan)
end

end
