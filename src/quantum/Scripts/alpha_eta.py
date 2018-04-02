#!/usr/bin/env python

import os
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt

from diff import relSpacing
from custom_parser import parse
from tools import find, get_input, cd
from alpha import find_max_e
from eta import compute_eta
import alpha
import eta


def main(B, D, N, max_energy=[0], energy_plot=False, small_plot=False):
    figsize = (5.8, 4) if not small_plot else (5.8, 3.5)
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=2)
    symblols = ['o', '^', 'v', 's', 'p', '*']
    if energy_plot:
        marks = itertools.cycle(symblols[:len(B)])
    else:
        marks = itertools.cycle(symblols[:len(max_energy)])

    # alpha
    files = find('alpha*.txt', '../output')
    msize = 8
    for n_i in N:
        if energy_plot:
            for b_i in B:
                marker = next(marks)
                alpha.e_plot(files, ax[0], n_i, b_i, msize, marker)
        else:
            for max_e in max_energy:
                marker = next(marks)
                alpha.b_plot(files, ax[0], n_i, max_e, msize, marker)
        msize -= 2

    # eta
    msize = 4
    for n_i in N:
        if energy_plot:
            for b_i in B:
                marker = next(marks)
                eta.e_plot(b_i, D[0], n_i, max_energy, ax[1], msize, marker)
        else:
            for max_e in max_energy:
                marker = next(marks)
                eta.b_plot(B, D[0], n_i, max_e, ax[1], msize, marker)
        msize -= 2

    ax[0].set_ylabel('$\\alpha$')
    ax[1].set_ylabel('$\\eta$')
    for col in ax:
        if energy_plot:
            col.set_xlabel('$\\Delta E$')
        else:
            col.set_xlabel('$B$')
        col.set_ylim([0, 1.2])
        if not energy_plot:
            col.set_xlim([0, 1])
        col.legend()
    plt.tight_layout(pad=0.3)
    # fig.show()
    if energy_plot:
        fig.savefig('../Statistics/alpha-eta_e_B[' +
                    ', '.join('{:.2}' for i in B).format(*B) +
                    ']_N' + str(N) + '.pdf', dpi=400)
    else:
        fig.savefig('../Statistics/alpha-eta_N' + str(N) +
                    ('_max_e_' + str(max_energy) + '.pdf'
                     if len(max_energy) > 1 or max_energy[0] else '.pdf'),
                    dpi=400)
    plt.close()


if __name__ == '__main__':
    B, D, N, max_energy, energy_plot, small_plot = \
        parse(max_e=True, e_plot=True, s_plot=True)
    main(B, D, N, max_energy, energy_plot, small_plot)
# %cd Scripts/
# %scope
