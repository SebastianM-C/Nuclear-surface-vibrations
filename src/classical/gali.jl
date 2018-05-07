module Gali

export galimap

include("hamiltonian.jl")
include("initial_conditions.jl")

using .Hamiltonian, .InitialConditions
using DynamicalSystems
using StaticArrays
using DataFrames, CSV

function galimap(E; A=1, B=0.55, D=0.4, n=15, m=15, tmax=500, dt=1, threshold=1e-12,
                 diff_eq_kwargs=Dict(:abstol=>1e-14, :reltol=>1e-14))
    prefix = "../../output/classical/B$B-D$D/E$E"
    q0, p0, N = generateInitialConditions(E, n, m, params=(A,B,D))
    df = CSV.read("$prefix/z0.csv", allowmissing=:none, use_mmap=!is_windows())
    # Workaround for https://github.com/JuliaData/CSV.jl/issues/170
    if !haskey(df, :gali)
        chaoticity = _galimap(q0, p0, N; A=A, B=B, D=D, tmax=tmax, dt=dt,
            threshold=threshold, diff_eq_kwargs=diff_eq_kwargs)
        df[:gali] = chaoticity
        CSV.write("$prefix/z0.csv", df)
    else
        df = CSV.read("$prefix/z0.csv", allowmissing=:none)
        chaoticity = df[:gali]
    end
    return chaoticity
end

function _galimap(q0, p0, N; A=1, B=0.55, D=0.4, tmax=500, dt=1, threshold=1e-12,
                  diff_eq_kwargs=Dict(:abstol=>1e-14, :reltol=>1e-14))
    z0 = [SVector{4}(hcat(p0[i, :], q0[i, :])) for i=1:N]
    ds = ContinuousDynamicalSystem(ż, z0[1], (A,B,D))
    tinteg = tangent_integrator(ds, 2, diff_eq_kwargs=diff_eq_kwargs)
    chaoticity = Vector{Float64}(N)

    for i=1:N
        set_state!(tinteg, z0[i])
        set_deviations!(tinteg, orthonormal(4, 2))
        reinit!(tinteg, tinteg.u)
        chaoticity[i] = ChaosTools._gali(tinteg, tmax, dt, threshold)[2][end]
    end
    return chaoticity
end

end  # module Gali
