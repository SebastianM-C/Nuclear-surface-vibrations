module Custom

export ScalarSavingPeriodicCallback

using DiffEqCallbacks, DiffEqBase, DataStructures

mutable struct ScalarSavingPeriodicAffect{AffectFunc, UpdateFunc, savevalType, tType}
    f::AffectFunc
    update_func::UpdateFunc
    saved_value::savevalType
    Δt::tType
    tnext::Ref{tType}
end

function (affect!::ScalarSavingPeriodicAffect)(integrator)
    affect!.update_func(Ref(affect!), integrator)
    affect!.f(integrator)
    # @show affect!.saved_value

    # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
    tnew = affect!.tnext[] + affect!.Δt
    tstops = integrator.opts.tstops
    for i in length(tstops) : -1 : 1 # reverse iterate to encounter large elements earlier
        if DataStructures.compare(tstops.comparer, tnew, tstops.valtree[i]) # TODO: relying on implementation details
            affect!.tnext[] = tnew
            add_tstop!(integrator, tnew)
            break
        end
    end
end

"""
    ScalarSavingPeriodicCallback(f, update_func, initial_val, Δt::Number;
                                    initialize = DiffEqBase.INITIALIZE_DEFAULT,
                                    initial_affect = true, kwargs...)
A `DiscreteCallback` applied with the period `Δt` and the value
of `save_func(u, t, integrator)` in `saved_values`.
If `save_everystep`, every step of the integrator is saved.
If `saveat` is specified, the values are saved at the given times, using
interpolation if necessary.
If the time `tdir` direction is not positive, i.e. `tspan[1] > tspan[2]`,
`tdir = -1` has to be specified.
"""
function ScalarSavingPeriodicCallback(f, update_func, initial_val, Δt::Number;
                                         initialize = DiffEqBase.INITIALIZE_DEFAULT,
                                         initial_affect = true, kwargs...)
    # Value of `t` at which `f` should be called next:
    # tnext = Ref(typemax(Δt))
    affect! = ScalarSavingPeriodicAffect(f, update_func, initial_val, Δt, Ref(typemax(Δt)))
    condition = (u, t, integrator) -> t == affect!.tnext[]


    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_periodic = function (c, u, t, integrator)
        @assert integrator.tdir == sign(Δt)
        initialize(c, u, t, integrator)
        if initial_affect
            affect!.tnext[] = t
            affect!(integrator)
        else
            affect!.tnext[] = time_choice(integrator)
            add_tstop!(integrator, affect!.tnext[])
        end
    end

    DiscreteCallback(condition, affect!; initialize = initialize_periodic, kwargs...)
end

end  # module Custom
