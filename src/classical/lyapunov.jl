nworkers() == 1 && addprocs(Int(Sys.CPU_CORES / 2))

!contains(==, names(Main), :Hamiltonian) && @everywhere include("hamiltonian.jl")
!contains(==, names(Main), :InitialConditions) && include("initial_conditions.jl")

using Hamiltonian, InitialConditions
@everywhere using DynamicalSystemsBase, ChaosTools
using StaticArrays
using DataFrames, CSV
using PmapProgressMeter

function λmap(E; A=1, B=0.55, D=0.4, n=15, m=15, T=10000., Ttr=1000., d0=1e-9,
               upper_threshold=1e-6, lower_threshold=1e-12,
               inittest = ChaosTools.inittest_default(4),
               dt=10, diff_eq_kwargs=Dict(:abstol=>1e-14, :reltol=>1e-14,
               :maxiters=>1e9))
    prefix = "../../output/classical/B$B-D$D/E$E"
    q0, p0, N = generateInitialConditions(E, n, m, params=(A,B,D))
    df = CSV.read("$prefix/z0.csv", allowmissing=:none, use_mmap=!is_windows())
    # Workaround for https://github.com/JuliaData/CSV.jl/issues/170
    if !haskey(df, :λs)
        λs = _λmap(q0, p0, N; A=A, B=B, D=D, T=T, Ttr=Ttr,
           d0=d0, upper_threshold=upper_threshold, lower_threshold=lower_threshold,
           inittest=inittest, dt=dt, diff_eq_kwargs=diff_eq_kwargs)
        df[:λs] = λs
        CSV.write("$prefix/z0.csv", df)
    else
        df = CSV.read("$prefix/z0.csv", allowmissing=:none)
        λs = df[:λs]
    end
    return λs
end

function _λmap(q0, p0, N; A=1, B=0.55, D=0.4, T=5000., Ttr=100., d0=1e-9,
               upper_threshold=1e-6, lower_threshold=1e-12,
               inittest = ChaosTools.inittest_default(4),
               dt=10., diff_eq_kwargs=Dict(:abstol=>d0, :reltol=>d0))
    z0 = [SVector{4}(hcat(p0[i, :], q0[i, :])) for i=1:N]
    ds = DynamicalSystemsBase.ContinuousDynamicalSystem(ż, z0[1], (A,B,D))

    pinteg = DynamicalSystemsBase.parallel_integrator(ds,
            [deepcopy(DynamicalSystemsBase.get_state(ds)),
            inittest(DynamicalSystemsBase.get_state(ds), d0)];
            diff_eq_kwargs=diff_eq_kwargs)
    λs = SharedArray{Float64}(N)

    pmap(i->(set_state!(pinteg, z0[i]);
            reinit!(pinteg, pinteg.u);
            λs[i] = ChaosTools._lyapunov(pinteg, T, Ttr, dt, d0,
                upper_threshold, lower_threshold)),
        PmapProgressMeter.Progress(N), 1:N)

    return Array(λs)
end
