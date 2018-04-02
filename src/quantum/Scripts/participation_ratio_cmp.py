#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

import eigensystem
from tools import cd
from custom_parser import parse
from participation_ratio import compute_p


def plot(E, P, E_ref, P_ref, label, fname, ylim=None):
    plt.scatter(E, P, s=1, label=label)
    plt.scatter(E_ref, P_ref, s=1, label='$B=0$')
    plt.xlabel('$E$')
    plt.ylabel('Participation ratio')
    ymin, ymax = plt.ylim()
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.savefig(fname)
    plt.close()
    if not ylim:
        return ymin, ymax


def main(b, d, n):
    cd(b, d, n)
    # Get states
    E, eigenvectors, ket = eigensystem.get(return_eigv=True, return_ket=True)
    # Select stable levels
    st_idx = int(np.loadtxt('stable.txt')[0])
    E, eigenvectors, ket = E[:st_idx], eigenvectors[:st_idx], ket[:st_idx]

    # Get reference states
    cd(0.0, d, n)
    E_ref, eigenvectors_ref, ket_ref = eigensystem.get(return_eigv=True,
                                                       return_ket=True)
    # Select stable reference levels
    st_idx = int(np.loadtxt('stable.txt')[0])
    E_ref, eigenvectors_ref, ket_ref = E_ref[:st_idx], \
        eigenvectors_ref[:st_idx], ket_ref[:st_idx]
    # Reference participation ratio
    P_ref = compute_p(eigenvectors_ref)
    # Find the index of the states with energy closest to the reference ones
    # idx = np.searchsorted(E, E_ref)
    # idx = np.clip(idx, 1, len(E) - 1)
    # left = E[idx - 1]
    # right = E[idx]
    # idx -= E_ref - left < right - E_ref
    # Compute the participation ratio
    P = compute_p(eigenvectors)

    cd(b, d, n)
    ylim = ()
    ylim = plot(E, P, E_ref, P_ref, label='$B=' + str(b) + '$',
                fname='participation_ratio_cmp.pdf')

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
    plot(rebde, P_b[::2], E_ref, P_ref, label='$\Gamma_b$', ylim=ylim,
         fname='participation_ratio_cmp_rebde.pdf')
    plot(reuna, P_a, E_ref, P_ref, label='$\Gamma_a$', ylim=ylim,
         fname='participation_ratio_cmp_reuna.pdf')
    plot(reuns, P_s, E_ref, P_ref, label='$\Gamma_s$', ylim=ylim,
         fname='participation_ratio_cmp_reuns.pdf')

    # plt.scatter(E_ref, P - P_ref, s=1)
    # plt.xlabel('$E$')
    # plt.ylabel('$\\Delta$Participation ratio')
    # plt.savefig('participation_ratio_diff.pdf')
    # plt.close()


if __name__ == '__main__':
    B, D, N = parse()
    for b in B:
        for d in D:
            for n in N:
                main(b, d, n)
