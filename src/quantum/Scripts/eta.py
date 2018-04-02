#!/usr/bin/env python

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from tools import cd
from custom_parser import parse
from diff import relSpacing


def compute_eta(rep):
    s = relSpacing(rep)                 # relative spacings
    σ = np.mean(s**2) - np.mean(s)**2   # varaince
    σ_W = 4 / np.pi - 1
    σ_P = 1
    η = (σ - σ_W) / (σ_P - σ_W)
    return η


def b_plot(B, d, n_i, max_e, ax, msize, marker):
    """Plot eta as a function of B"""
    reps = 'reuna', 'reuns', 'rebde'
    rnames = {'reuna': r'$\Gamma_a$', 'reuns': r'$\Gamma_s$',
              'rebde': r'$\Gamma_b$'}
    for r in reps:
        values = []
        for b in B:
            cd(b, d, n_i)
            rep = np.loadtxt(r + '.dat', usecols=(0,))
            values.append(compute_eta(rep[rep < max_e] if max_e else rep))
            os.chdir('../../Scripts')
        # Plot the results
        ax.plot(B, values, linestyle='', label=rnames[r], markersize=msize,
                marker=marker)


def e_plot(b_i, d, n_i, max_energy, ax, msize, marker):
    """Plot eta as a function of deltaE"""
    cd(b_i, d, n_i)
    reps = 'reuna', 'reuns', 'rebde'
    rnames = {'reuna': r'$\Gamma_a$', 'reuns': r'$\Gamma_s$',
              'rebde': r'$\Gamma_b$'}
    for r in reps:
        values = []
        rep = np.loadtxt(r + '.dat', usecols=(0,))
        for max_e in max_energy:
            values.append(compute_eta(rep[rep < max_e] if max_e else rep))
        # Plot the results
        max_energy = np.array(max_energy)
        max_energy[max_energy == 0] = rep[-1] - rep[0]
        ax.plot(max_energy, values, linestyle='',
                label=r'$B=' + str(b_i) + r'$, ' + rnames[r],
                markersize=msize, marker=marker)
    os.chdir('../../Scripts')


def main(B, D, N, max_energy=[0], energy_plot=False, small_plot=False):
    figsize = (5.8, 4) if not small_plot else (5.8, 3.5)
    fig, ax = plt.subplots(figsize=figsize)
    symblols = ['o', '^', 'v', 's', 'p', '*']
    if energy_plot:
        marks = itertools.cycle(symblols[:len(B)])
    else:
        marks = itertools.cycle(symblols[:len(max_energy)])
    msize = 4
    for n_i in N:
        if energy_plot:
            for b_i in B:
                marker = next(marks)
                e_plot(b_i, D[0], n_i, max_energy, ax, msize, marker)
        else:
            for max_e in max_energy:
                marker = next(marks)
                b_plot(B, D[0], n_i, max_e, ax, msize, marker)
        msize -= 2

    ax.set_ylabel('$\\eta$')
    if energy_plot:
        ax.set_xlabel('$\\Delta E$')
    else:
        ax.set_xlabel('$B$')
    ax.set_ylim([0, 1.2])
    if not energy_plot:
        ax.set_xlim([0, 1])
    ax.legend()
    # plt.show()
    plt.tight_layout(pad=0.3)
    if energy_plot:
        fig.savefig('../Statistics/eta_e_B[' +
                    ', '.join('{:.2}' for i in B).format(*B) +
                    ']_N' + str(N) + '.pdf', dpi=400)
    else:
        fig.savefig('../Statistics/eta_N' + str(N) +
                    ('_max_e_' + str(max_energy) + '.pdf'
                     if len(max_energy) > 1 or max_energy[0] else '.pdf'),
                    dpi=400)
    plt.close()


if __name__ == '__main__':
    B, D, N, max_energy, energy_plot, small_plot = \
        parse(max_e=True, e_plot=True, s_plot=True)
    main(B, D, N, max_energy, energy_plot, small_plot)
