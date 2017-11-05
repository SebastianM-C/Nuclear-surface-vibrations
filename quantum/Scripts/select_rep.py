#!/usr/bin/env python

import os
import numpy as np
from timeit import default_timer as timer

import eigensystem
import diff
from custom_parser import parse
from tools import cd


def main(b, d, n, delta_n, st_epsilon, lvl_epsilon, cut=0):
    print("Running with: B = " + str(b) + " D = " + str(d) + " N = " + str(n))
    cd(b, d, n)
    start = timer()
    E, ket = eigensystem.get(return_ket=True)
    # Select irreducible representations
    # ir_reps = eigensystem.levels(E, ket)

    # Select only the stable levels
    # stable_levels = diff.stable(E, ir_reps, b, d, n, delta_n)
    stable_levels = diff.stable(E, b, d, n, delta_n, st_epsilon)
    # Reduce the number of stable levels to check the convergence
    stable_levels = int((1 - cut) * stable_levels)
    E = E[:stable_levels]
    # Select irreducible representations
    ir_reps = eigensystem.levels(E, ket, lvl_epsilon)

    stop = timer()
    print('Get data:', stop - start, 'seconds')
    rebde = open('rebde.dat', 'w')
    reuna = open('reuna.dat', 'w')
    reuns = open('reuns.dat', 'w')

    # Write only one level corresponding to the bidimensional representation
    skip_next = False
    for i in range(E.size):
        n1 = ket[i][0]
        n2 = ket[i][1]
        if ir_reps[i] == 2:     # bidimensional representation (rebde)
            if not skip_next:
                rebde.write('{0:.18f}'.format(E[i]) + '\t' + str(n1) + '\t' +
                            str(n2) + '\n')
                skip_next = True
            else:
                skip_next = False
        else:
            if n2 % 2:          # unidimensional anti-symmetric representation
                reuna.write('{0:.18f}'.format(E[i]) + '\t' + str(n1) + '\t' +
                            str(n2) + '\n')
            else:               # unidimensional symmetric representation
                reuns.write('{0:.18f}'.format(E[i]) + '\t' + str(n1) + '\t' +
                            str(n2) + '\n')

    rebde.close()
    reuna.close()
    reuns.close()

    os.chdir("../../Scripts")
    print("Done\n")


if __name__ == '__main__':
    B, D, N, delta_n, st_epsilon, lvl_epsilon = parse(advanced=True)

    for b in B:
        for d in D:
            for n in N:
                main(b, d, n, delta_n, st_epsilon, lvl_epsilon)
