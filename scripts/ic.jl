using NuclearSurfaceVibrations
using .Classical

E = 10.
q0, p0 = initial_conditions(E, alg=PoincareRand(n=10))

λmap(E, ic_alg=PoincareRand(n=10))
