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
    fig, ax = plt.subplots(figsize=figsize)
    d = D[0]
    # alpha
    msize = 8
    reps = 'reuna', 'reuns', 'rebde'
    for r in reps:
        values = []
        for n_i in N:
            for max_e in max_energy:
                for b_i in B:
                    cd(b_i, d, n_i)
                    rep = np.loadtxt(r + '.dat', usecols=(0,))
                    files = find('alpha*.txt', '.')
                    for f in files:
                        regex = r"""alpha_max_e_([0-9]+)"""
                        f_max_e = re.compile(regex).search(f)
                        if f_max_e:
                            f_max_e = float(f_max_e.group(1))
                        else:
                            f_max_e = 0
                        if max_e == f_max_e:
                            print(max_e, f_max_e, f)
                            values.append([np.asscalar(np.loadtxt(f)),
                                compute_eta(rep[rep < max_e] if max_e else rep)])
        ax.scatter(np.array(values)[:, 0], np.array(values)[:, 1])


    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$\\eta$')
    fig.savefig('../../../Statistics/correlation_B[' +
                ', '.join('{:.2}' for i in B).format(*B) +
                ']_N' + str(N) + '.pdf', dpi=400)


if __name__ == '__main__':
    B, D, N, max_energy, energy_plot, small_plot = \
        parse(max_e=True, e_plot=True, s_plot=True)
    main(B, D, N, max_energy, energy_plot, small_plot)
# %cd Scripts/
# %scope
