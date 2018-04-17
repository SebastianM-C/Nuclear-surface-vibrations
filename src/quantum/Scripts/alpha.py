#!/usr/bin/env python

import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
from custom_parser import parse

from tools import find, get_input


def find_max_e(f):
    regex = r"""alpha_max_e_([0-9]+)"""
    max_e = re.compile(regex).search(f)
    if max_e:
        max_e = float(max_e.group(1))
    else:
        max_e = np.loadtxt(f.replace('alpha', 'stable'))[1]
    return max_e


def b_plot(files, ax, n_i, max_energy, msize, marker):
    """Plot alpha as a function of B"""
    values = []
    count = 0
    avg_deltaE = 0
    for f in files:
        b, d, n = get_input(f)
        max_e = find_max_e(f) if max_energy else 0
        if not max_energy:
            regex = r"""alpha_max_e_"""
            if re.compile(regex).search(f):
                continue
        if int(n) == n_i and max_energy == max_e:
            values.append([b, np.asscalar(np.loadtxt(f))])
            if max_energy:
                avg_deltaE += max_e
            else:
                avg_deltaE += np.loadtxt(f.replace('alpha', 'stable'))[1]
            count += 1
    if not count:
        print('No values for N =', n_i, 'and max energy', max_energy)
        return
    avg_deltaE = avg_deltaE / count
    print('Found', count, 'values for N =', n_i,
          'and avg deltaE', avg_deltaE)
    # Plot the results
    ax.plot(np.array(values)[:, 0], np.array(values)[:, 1], linestyle='',
            label=r'$\Delta E\approx' + '{:.2f}'.format(avg_deltaE) +
            ', N=' + str(n_i) + '$', markersize=msize, marker=marker)


def e_plot(files, ax, n_i, b_i, msize, marker):
    """Plot alpha as a function of deltaE"""
    values = []
    count = 0
    for f in files:
        b, d, n = get_input(f)
        if n == n_i and b == b_i:
            max_e = find_max_e(f)
            values.append((max_e, np.asscalar(np.loadtxt(f))))
            count += 1
    if not count:
        print('No values for N =', n_i, 'and B =', b_i)
        return
    print('Found', count, 'values for N =', n_i, 'and B =', b_i)
    # Plot the results
    ax.plot(np.array(values)[:, 0], np.array(values)[:, 1], linestyle='',
            label=r'$B=' + str(b_i) + '$', markersize=msize, marker=marker)


def main(B, D, N, max_energy=[0], energy_plot=False, small_plot=False):
    figsize = (5.8, 4) if not small_plot else (5.8, 3.5)
    fig, ax = plt.subplots(figsize=figsize)
    symblols = ['o', '^', 'v', 's', 'p', '*']
    if energy_plot:
        marks = itertools.cycle(symblols[:len(B)])
    else:
        marks = itertools.cycle(symblols[:len(max_energy)])
    files = find('alpha*.txt', '../Output')
    msize = 8
    for n_i in N:
        if energy_plot:
            for b_i in B:
                marker = next(marks)
                e_plot(files, ax, n_i, b_i, msize, marker)
        else:
            for max_e in max_energy:
                marker = next(marks)
                b_plot(files, ax, n_i, max_e, msize, marker)
        msize -= 2

    ax.set_ylabel('$\\alpha$')
    if energy_plot:
        ax.set_xlabel('$\\Delta E$')
    else:
        ax.set_xlabel('$B$')
    ax.set_ylim([0, 1.1])
    if not energy_plot:
        ax.set_xlim([0, 1])
    ax.legend()
    # plt.show()
    plt.tight_layout(pad=0.3)
    if energy_plot:
        fig.savefig('../Statistics/alpha_e_B[' +
                    ', '.join('{:.2}' for i in B).format(*B) +
                    ']_N' + str(N) + '.pdf', dpi=400)
    else:
        fig.savefig('../Statistics/alpha_N' + str(N) +
                    ('_max_e_' + str(max_energy) + '.pdf'
                     if len(max_energy) > 1 or max_energy[0] else '.pdf'),
                    dpi=400)
    plt.close()


if __name__ == '__main__':
    B, D, N, max_energy, energy_plot, small_plot = \
        parse(max_e=True, e_plot=True, s_plot=True)
    main(B, D, N, max_energy, energy_plot, small_plot)
