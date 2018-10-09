module ParallelTrajectories

export parallel_problem, create_parallel

using ..Distributed

using StaticArrays
using OrdinaryDiffEq
# using ChaosTools

function create_parallel(f, u)
    u₀ = vcat(u...)

    @inbounds function eom(u::SVector{N}, p, t) where {N}
        idx1 = SVector{N÷2}(1:N÷2)
        idx2 = SVector{N÷2}(N÷2+1:N)
        vcat(f(u[idx1], p, t),
             f(u[idx2], p, t))
    end

    return eom, u₀
end

function create_parallel(ṗ, q̇, p0, q0)
    p₀ = vcat(p0...)
    q₀ = vcat(q0...)

    @inbounds function f1(p::SVector{N}, q::SVector{N}, params, t) where {N}
        idx1 = SVector{N÷2}(1:N÷2)
        idx2 = SVector{N÷2}(N÷2+1:N)
        vcat(ṗ(p[idx1], q[idx1], params, t),
             ṗ(p[idx2], q[idx2], params, t))
    end

    @inbounds function f2(p::SVector{N}, q::SVector{N}, params, t) where {N}
        idx1 = SVector{N÷2}(1:N÷2)
        idx2 = SVector{N÷2}(N÷2+1:N)
        vcat(q̇(p[idx1], q[idx1], params, t),
             q̇(p[idx2], q[idx2], params, t))
    end

    return f1, f2, p₀, q₀
end

function parallel_problem(f, u, tspan, params, cb=nothing)
    peom, u₀ = create_parallel(f, u)
    ODEProblem(peom, u₀, tspan, params, callback=cb)
end

function parallel_problem(ṗ, q̇, p0, q0, tspan, params, cb=nothing)
    f1, f2, p₀, q₀ = create_parallel(ṗ, q̇, p0, q0)
    DynamicalODEProblem(f1, f2, p₀, q₀, tspan, params, callback=cb)
end

# function parallel_problem(ds::ChaosTools.CDS, states, tspan)
#     peom, st = DynamicalSystemsBase.create_parallel(ds, states)
#     ODEProblem(peom, st, tspan, ds.p)
# end

end  # module ParallelTrajectories
