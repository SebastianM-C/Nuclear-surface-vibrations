# !/usr/bin/env python

import os
import numpy as np
import eigensystem

import tools
from plots import bar_plot, histogram


def relSpacing(E):
    deltaE = np.diff(E)
    avgSpacing = (E[-1] - E[0]) / E.size
    return deltaE / avgSpacing


def difference(E1, b, d, n, delta_n, ir_reps1=np.empty(0)):
    """Return the differences between the energy levels and of the
    structure of the irreducible representations (optional) for the given
    diagonalization basis and n + delta_n"""
    # Load the energy levels in the second basis
    os.chdir("../B" + str(b) + " D" + str(d) + " N" + str(n + delta_n))
    E2, ket2 = eigensystem.get(return_ket=True)
    if ir_reps1.size:
        ir_reps2 = eigensystem.levels(E2, ket2)

    if E2.size > E1.size:
        E_diff = (E2[:E1.size] - E1) / E1
        if ir_reps1.size:
            ir_diff = ir_reps2[:ir_reps1.size] - ir_reps1
    else:
        E_diff = (E1[:E2.size] - E2) / E2
        if ir_reps1.size:
            ir_diff = ir_reps1[:ir_reps2.size] - ir_reps2

    os.chdir("../B" + str(b) + " D" + str(d) + " N" + str(n))
    if ir_reps1.size == 0:
        return np.abs(E_diff)
    return np.abs(E_diff), ir_diff


def stable(E1, b, d, n, delta_n, epsilon, ir_reps=np.empty(0)):
    """Return the number of stable levels"""
    if ir_reps.size == 0:
        E_diff = difference(E1, b, d, n, delta_n)
    else:
        E_diff, ir_diff = difference(E1, b, d, n, delta_n, ir_reps)
    # Energy difference (between two diagonalization bases) histogram
    histogram(E_diff, label='B' + str(b) + ' D' + str(d) + ' N' + str(n),
              bins=np.pad(np.logspace(-14, -2, 13), (1, 0), mode='constant'),
              xscale='log', fname='hist_E_diff.pdf')
    # Energy difference bar plot
    bar_plot(E_diff[E_diff < 0.01],
             label=r'$\delta_s = 10^{-9}$', ylabel=r'$E_{N+ \Delta N} - E_N$',
             figsize=(5.8, 4), axhline_y=epsilon, yscale='log', dpi=600,
             fname='bar_E_diff.pdf', xlabel='index')

    last_stable = np.where(E_diff > epsilon)[0][1]
    # Workaround for the initial instability in some particular cases
    if last_stable < 10:
        last_stable = np.where(E_diff > epsilon)[0][3]
    # Cache the result
    np.array([last_stable, E1[last_stable] - E1[0]]).tofile('stable.txt',
                                                            sep='\n')
    print("E_diff > epsilon :",
          np.where(E_diff > epsilon)[0][:5], "\nstability epsilon: ", epsilon)
    print("E 0 =", E1[0], "\nE", last_stable, "=", E1[last_stable])
    if ir_reps.size:
        print("ir_diff: ", np.where(ir_diff > 0)[0][:5])
    return last_stable
