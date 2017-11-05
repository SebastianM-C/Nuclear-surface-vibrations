#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from custom_parser import parse

from tools import find, get_input


def main(B, D, N):
    files = find('avg_sp.txt', '../Output')
    fig, ax = plt.subplots(figsize=(5.8, 4))
    msize = 8
    for n_i in N:
        values = []
        count = 0
        for f in files:
            b, d, n = get_input(f)
            if int(n) == n_i:
                values.append([b, np.loadtxt(f)])
                count += 1
        print('Found', count, 'values for N =', n_i)
        results = np.zeros(count, [('b', np.float64),
                                   ('avg_sp', np.float64, 3)])

        for i in range(count):
            results[i] = tuple(values[i])
        # Plot the results
        ax.plot(results['b'], results['avg_sp'][:, 0], '^',
                label=r'$\Gamma_a,\;N=' + str(n_i) + '$', markersize=msize)
        ax.plot(results['b'], results['avg_sp'][:, 1], 'v',
                label=r'$\Gamma_s,\;N=' + str(n_i) + '$', markersize=msize)
        ax.plot(results['b'], results['avg_sp'][:, 2], 'o',
                label=r'$\Gamma_b,\;N=' + str(n_i) + '$', markersize=msize)
        msize -= 2
    ax.set_ylabel(r'$\overline{\Delta E_r}$')
    ax.set_xlabel('$B$')
    ax.set_xlim([0, 1])
    ax.legend()
    plt.tight_layout(pad=0.3)
    fig.savefig('../Statistics/avg_sp_N' + str(N) + '.pdf', dpi=200)
    plt.close()


if __name__ == '__main__':
    B, D, N = parse()
    main(B, D, N)
