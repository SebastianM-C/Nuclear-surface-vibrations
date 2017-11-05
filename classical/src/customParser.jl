#!/usr/bin/env julia
module CustomParser
using ArgParse

export getArguments

function getArguments(str; e_only=false)
  s = ArgParseSettings()
  @add_arg_table s begin
    "--energy", "-e"
        help = "The energy of the system"
        arg_type = Float64
        default = 30.
    "-n"
        help = "Number of initial conditions"
        arg_type = Int64
        default = 10
    "--tmax", "-t"
        help = "Simulation time for one initial condition"
        arg_type = Float64
        default = 1.e3
  end
  parsedArgs = parse_args(str, s)
  if e_only
      return parsedArgs["energy"]
  end
  parsedArgs["energy"], parsedArgs["n"], parsedArgs["tmax"]
end

end
