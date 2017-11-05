#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import eigensystem
from tools import cd
from custom_parser import parse


def compute_p(eigenvectors, condition=None):
    """Compute participation ratio"""
    if condition:
        return 1 / (eigenvectors[0].size *
                    np.sum(eigenvectors[condition]**4, axis=1))
    return 1 / (eigenvectors[0].size * np.sum(eigenvectors**4, axis=1))


def main(b, d, n):
    cd(b, d, n)
    # Get states
    E, eigenvectors, ket = eigensystem.get(return_eigv=True, return_ket=True)
    # Select stable levels
    st_idx = int(np.loadtxt('stable.txt')[0])
    E, eigenvectors, ket = E[:st_idx], eigenvectors[:st_idx], ket[:st_idx]
    if b:
        # Get irreducible representations
        ir_reps = eigensystem.levels(E, ket)
        rebde = np.loadtxt('rebde.dat', usecols=(0,))
        reuna = np.loadtxt('reuna.dat', usecols=(0,))
        reuns = np.loadtxt('reuns.dat', usecols=(0,))
        # Compute participation ratio for each representation
        P_b = compute_p(eigenvectors, condition=np.where(ir_reps == 2))
        P_a = compute_p(eigenvectors, condition=np.where(ir_reps == 1))
        P_s = compute_p(eigenvectors, condition=np.where(ir_reps == 0))
        # Plot the participation ratio for each representation
        plt.scatter(rebde, P_b[::2], s=1, label='$\Gamma_b$')
        plt.scatter(reuna, P_a, s=1, label='$\Gamma_a$', color='r')
        plt.scatter(reuns, P_s, s=1, label='$\Gamma_s$', color='y')
        plt.xlabel('$E$')
        plt.ylabel('Participation ratio')
        plt.legend()
        plt.savefig('participation_ratio_rep.pdf')
        plt.close()
        # Plot the difference between the states of the bidimensional
        # representation
        plt.plot(rebde, P_b[::2] - P_b[1::2], lw=0.7,
                 label='$\Gamma_{b1} - \Gamma_{b2}$')
        plt.xlabel('$E$')
        plt.ylabel('$\\Delta$Participation ratio')
        plt.legend()
        plt.savefig('participation_ratio_rebde.pdf')
        plt.close()
    else:
        # P = 1 / (eigenvectors[0].size * np.sum(eigenvectors**4, axis=1))
        P = compute_p(eigenvectors)
        plt.scatter(E, P, s=1)
        plt.xlabel('$E$')
        plt.ylabel('Participation ratio')
        plt.savefig('participation_ratio.pdf')
        plt.close()


if __name__ == '__main__':
    B, D, N = parse()
    for b in B:
        for d in D:
            for n in N:
                main(b, d, n)
