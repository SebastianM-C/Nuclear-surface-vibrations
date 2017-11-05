#!/usr/bin/env python

import os
import numpy as np

import select_rep
from tools import cd
from custom_parser import parse
from diff import relSpacing
from plots import bar_plot, histogram


def main(b, d, n, delta_n, st_epsilon, lvl_epsilon, reselect=True, cut=0,
         bin_size=0.25, max_energy=0):
    if reselect:
        select_rep.main(b, d, n, delta_n, st_epsilon, lvl_epsilon, cut)
    reps = 'reuna', 'reuns', 'rebde'
    rnames = {'reuna': r'$\Gamma_a$', 'reuns': r'$\Gamma_s$',
              'rebde': r'$\Gamma_b$'}
    cd(b, d, n)
    if max_energy:
        deltaE = max_energy
    else:
        deltaE = np.loadtxt('stable.txt')[1]
    count = int(4 / bin_size) + 1
    rel_sp = []     # relative spacings
    avg_sp = []     # average spacings
    w = []          # weights
    for r in reps:
        rep = np.loadtxt(r + '.dat', usecols=(0,))
        if max_energy:
            rep = rep[rep <= max_energy]
        rel_sp.append(relSpacing(rep))
        avg_sp.append((rep[-1] - rep[0]) / rep.size)
        # P(s)Δs is the probability, P(s) is the probability density
        # P(s)Δs = Σ 1/3 * N_rep/N_tot
        w.append(np.ones(rel_sp[-1].shape) / (3 * rep.size * bin_size))
        # Don't plot if not all levels are used
        if not max_energy:
            fname = r + '.pdf'
            histogram(rel_sp[-1], bins=np.linspace(0, 4, count), fname=fname,
                      label=rnames[r], xlabel='$s$', figsize=(2.8, 3),
                      ylabel='No. of levels')
            fname = 'bar_' + r + '.pdf'
            bar_plot(rel_sp[-1], label=rnames[r], ylabel='s', xlabel='index',
                     fname=fname, dpi=400, figsize=(2.8, 3),
                     title=r'$\frac{E_n-E_0}{N}=' +
                     '{:.3}'.format(avg_sp[-1]) + '$')

    # Save the average spacings
    fname = 'avg_sp' + \
        ('_max_e_' + str(max_energy) + '.txt' if max_energy else '.txt')
    with open(fname, 'w') as f:
        f.write('\n'.join([str(i) for i in avg_sp]))
    # Relative spacing histogram, P(s)
    # Don't plot if not all levels are used
    if not max_energy:
        fname = 'P(s)' + '_st_' + '{:.0e}'.format(st_epsilon) + '_eps_' + \
            '{:.0e}'.format(lvl_epsilon) + \
            ('_cut_' + '{:.2f}'.format(cut) + '.pdf' if cut else '.pdf')
        histogram(rel_sp, bins=np.linspace(0, 4, count), weights=w,
                  label=[rnames[i] for i in reps], fname=fname, stacked=True,
                  ylabel='$P(s)$', xlabel='$s$', count=count, use_wigner=True,
                  use_poisson=True, figsize=(5.8, 4.5))
    # Fitted P(s)
    fname = 'P(s)_fit_' + '{:.0e}'.format(st_epsilon) + '_eps_' + \
        '{:.0e}'.format(lvl_epsilon) + \
        ('_cut_' + '{:.2f}'.format(cut) if cut else '') + \
        ('_max_e_' + str(max_energy) + '.pdf' if max_energy else '.pdf')
    histogram(rel_sp, bins=np.linspace(0, 4, count), weights=w, ylim=(0, 1.05),
              title='$\\Delta E =' + '{:.5}'.format(deltaE) + '$',
              fname=fname, count=count, ylabel='$P(s)$', xlabel='$s$',
              stacked=True, label=[rnames[i] for i in reps], fit=True,
              max_e=max_energy, figsize=(5.8, 3.7))
    # For the cumulative distribution plots, each irreducible representation
    # is plotted individually (*3)
    # I(s) = Σ P(s) Δs = Σ Σ N_rep/N_tot * 1/Δs * Δs = Σ Σ N_rep/N_tot
    # (*bin_size)
    w = [wi * bin_size * 3 for wi in w]
    # cumulative relative spacing histogram, I(s)
    # Don't plot if not all levels are used
    if not max_energy:
        fname = 'I(s).pdf'
        histogram(rel_sp, cumulative=True, bins=np.linspace(0, 4, count),
                  ylabel=r'$I(s)$', xlabel='$s$', fname=fname, weights=w,
                  label=[rnames[i] for i in reps], count=count,
                  use_wigner=True, use_poisson=True, figsize=(5.8, 4.5))
    # Fitted I(s)
    fname = 'I(s)_fit' + \
        ('_max_e_' + str(max_energy) + '.pdf' if max_energy else '.pdf')
    histogram(rel_sp, cumulative=True, bins=np.linspace(0, 4, count), fit=True,
              ylabel=r'$I(s)$', xlabel='$s$', weights=w, count=count,
              label=[rnames[i] for i in reps], fname=fname, ylim=(0, 1.05),
              figsize=(5.8, 3.7))
    # Version
    with open('version.txt', 'w') as f:
        f.write('1.4.2')
    os.chdir("../../Scripts")


if __name__ == '__main__':
    B, D, N, delta_n, st_epsilon, lvl_epsilon, reselect, cut, bin_size, \
        max_energy = parse(advanced=True, select=True, hist_bin=True,
                           max_e=True)

    for b in B:
        for d in D:
            for n in N:
                for max_e in max_energy:
                    main(b, d, n, delta_n, st_epsilon, lvl_epsilon, reselect,
                         cut, bin_size, max_e)
