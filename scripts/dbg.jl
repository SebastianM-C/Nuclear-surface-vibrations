using Logging, LoggingExtras

function module_filter((level, _module, group, id))
    Base.moduleroot(_module) == NuclearSurfaceVibrations
end

dbg = EarlyFilteredLogger(module_filter, ConsoleLogger(stdout, Logging.Debug))
