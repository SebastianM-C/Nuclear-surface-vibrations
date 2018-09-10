module ParallelTrajectories

export dist_prob

using ..Distributed

using StaticArrays
using OrdinaryDiffEq

function create_parallel(f, p, u0::SVector{N, T}, d0) where {N, T}
    u1 = u0 + d0/√N
    u₀ = vcat(u0, u1)

    return eom, u₀
end

function create_parallel(ṗ, q̇, p0::SVector{N, T}, q0::SVector{N, T}, params, d0) where {N, T}
    p1 = p0 + d0/√N
    q1 = q0 + d0/√N
    p₀ = vcat(p0, p1)
    q₀ = vcat(q0, q1)

    @inbounds @inline function f1(p::SVector{N, T}, q::SVector{N, T}, params, t) where {N, T}
        idx1 = SVector{N÷2}(1:N÷2)
        idx2 = SVector{N÷2}(N÷2+1:N)
        vcat(ṗ(p[idx1], q[idx1], params, t),
             ṗ(p[idx2], q[idx2], params, t))
    end

    @inbounds @inline function f2(p::SVector{N, T}, q::SVector{N, T}, params, t) where {N, T}
        idx1 = SVector{N÷2}(1:N÷2)
        idx2 = SVector{N÷2}(N÷2+1:N)
        vcat(q̇(p[idx1], q[idx1], params, t),
             q̇(p[idx2], q[idx2], params, t))
    end

    return f1, f2, p₀, q₀
end

@inbounds @inline function eom(f, u::SVector{N, T}, p, t) where {N, T}
    idx1 = SVector{N÷2}(1:N÷+2)
    idx2 = SVector{N÷2}(N÷2+1:N)
    vcat(f(u[idx1], p, t),
         f(u[idx2], p, t))
end

function dist_prob(f, p, u0, d0, tspan, cb=nothing)
    d_eom, u₀ = create_parallel(f, p, u0, d0)
    ODEProblem(d_eom, u₀, tspan, p, callback=cb)
end

function dist_prob(ṗ, q̇, p0::SVector, q0::SVector, params, d0, tspan, cb=nothing)
    f1, f2, p₀, q₀ = create_parallel(ṗ, q̇, p0, q0, params, d0)
    DynamicalODEProblem(f1, f2, p₀, q₀, tspan, p, callback=cb)
end

end  # module ParallelTrajectories
