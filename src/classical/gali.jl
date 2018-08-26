nworkers() == 1 && addprocs(Int(Sys.CPU_CORES / 2))

!contains(==, names(Main), :Hamiltonian) && @everywhere include("hamiltonian.jl")
!contains(==, names(Main), :InitialConditions) && include("initial_conditions.jl")

using Hamiltonian, InitialConditions
@everywhere using DynamicalSystemsBase, ChaosTools
using StaticArrays
using DataFrames, CSV
using PmapProgressMeter

function galimap(E; A=1, B=0.55, D=0.4, n=15, m=15, tmax=500, dt=1, threshold=1e-12,
                 diff_eq_kwargs=Dict(:abstol=>1e-14, :reltol=>1e-14), recompute=false)
    prefix = "../../output/classical/B$B-D$D/E$E"
    q0, p0, N = initial_conditions(E, n, m, params=(A,B,D))
    df = CSV.read("$prefix/z0.csv", allowmissing=:none, use_mmap=!is_windows())
    # Workaround for https://github.com/JuliaData/CSV.jl/issues/170
    if !haskey(df, :gali) || recompute
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
    ds = DynamicalSystemsBase.ContinuousDynamicalSystem(ż, z0[1], (A,B,D))
    tinteg = DynamicalSystemsBase.tangent_integrator(ds, 2, diff_eq_kwargs=diff_eq_kwargs)
    chaoticity = SharedArray{Float64}(N)

    pmap(i->(set_state!(tinteg, z0[i]);
            set_deviations!(tinteg, orthonormal(4, 2));
            reinit!(tinteg, tinteg.u);
            chaoticity[i] = ChaosTools._gali(tinteg, tmax, dt, threshold)[2][end]),
        PmapProgressMeter.Progress(N), 1:N)

    return Array(chaoticity)
end
