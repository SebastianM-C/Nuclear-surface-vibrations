#!/usr/bin/env python

import logging
import numpy as np
from scipy import linalg
from timeit import default_timer as timer
from os.path import isfile
from os import remove

from tools import get_input, latex_float
# from hamiltonian import main as hamiltonian
from plots import bar_plot, histogram


def readH(format):
    """Read the Hamiltonian using the given format"""
    if format == 'npz':
        if isfile('hamilt.npz'):
            hamilt = np.load('hamilt.npz')
            return hamilt['H']
        else:
            # Fallback to Fortran binary
            format = 'fortran_bin'
            # print('Hamiltonian file not found. Computing again.')
            # b, d, n = get_input()
            # return hamiltonian(1, b, d, n)
    if format == 'fortran_bin':
        _, _, n = get_input()
        nn = int(n * (n + 1) / 2)
        H = np.empty((nn, nn))
        with open('hamilt.bin', 'rb') as h:
            for i in range(nn):
                H[i] = np.fromfile(h, dtype='float64', count=nn).reshape(nn)
        return H
    # if format == 'text':
    #     H = np.loadtxt("hamilt.out")
    #     return H.T


def get(return_eigv=False, return_ket=False, return_index=False,
        return_cmax=False, return_H=False):
    """Return the eigenvalues and optionally the eigenvectors,
    the number operator form of the states(ket), the state index of the states,
    the max coefficient index and the Hamiltonian"""
    # Load files
    H = readH('npz')    # read the Hamiltonian
    # Save to npz to save sapce
    if not isfile('hamilt.npz'):
        np.savez_compressed('hamilt.npz', H=H)
        remove('hamilt.bin')
    b, d, n = get_input()
    n = int(n)
    index = np.array([(n1, n2) for n1 in range(n) for n2 in range(n - n1)])
    # Get eigenvalues and eigenvectors
    if isfile('eigensystem.npz'):
        print('Used cached result for: B =', b, 'D =', d, 'N =', n)
        eigensystem = np.load('eigensystem.npz')
        E = eigensystem['E']
        eigenvectors = eigensystem['eigenvectors']
    else:
        start = timer()
        E, eigenvectors = linalg.eigh(H, turbo=True)
        end = timer()
        print('Diagonalisation for N =', n, ':', end - start, 'seconds')
        # Save the results
        np.savez_compressed('eigensystem.npz', E=E, eigenvectors=eigenvectors)

    eigenvectors = np.transpose(eigenvectors)  # each eigenvector is on one row

    # max coefficient in eigenvector
    c_max = np.empty(eigenvectors.shape[0], dtype=int)

    # The index of the largest coefficient
    for i in range(eigenvectors.shape[0]):
        c_max[i] = np.argmax(np.abs(eigenvectors[i]))

    results = (E, )
    if return_eigv:
        results += (eigenvectors, )
    if return_ket:
        results += (index[c_max], )
    if return_index:
        results += (index, )
    if return_cmax:
        results += (c_max, )
    if return_H:
        results += (H, )
    return results


def levels(E, ket, epsilon=1e-8, colors=''):
    """Return the degenerate subspace index and optionally the colormap"""
    # irreducible representations
    # 0 - unidimensional symmetric representation (reuns)
    # 1 - unidimensional anti-symmetric representation (reuna)
    # 2 - bidimensional representation (rebde)
    ir_reps = np.zeros([E.size], dtype=np.uint8)
    return_colors = len(colors)
    if return_colors:
        colormap = [''] * E.size   # colors used

    # Group energy levels such that a level contains all the eigenvalues with
    # the same value
    delta = np.diff(E)
    avgSpacing = (E[-1] - E[0]) / E.size
    relsp = delta / avgSpacing
    print('levels epsilon:', epsilon)
    print('avgSpacing:', avgSpacing)

    levels = np.split(E, np.where(relsp > epsilon)[0] + 1)
    states = np.split(ket, np.where(relsp > epsilon)[0] + 1)

    # Energy difference (between two consecutive levels) histogram
    histogram(delta, xlabel=r'$\Delta E$', xscale='log',
              bins=np.pad(np.logspace(-15, 1, 17), (1, 0),
                          mode='constant'), ylabel='No. of levels',
              fname='hist_delta.pdf', figsize=(5.8, 3))
    # Relative spacing histogram
    histogram(relsp, xscale='log', ylabel='No. of levels',
              bins=np.pad(np.logspace(-13, 1, 15), (1, 0),
                          mode='constant'), fname='hist_relsp.pdf',
              xlabel='$s$', figsize=(2.8, 3))
    # Energy difference bar plot
    bar_plot(delta, figsize=(5.8, 3), ylabel=r'$\Delta E$', yscale='log',
             xlabel='index', fname='bar_delta.pdf', dpi=720)
    # Relative spacing bar plot
    bar_plot(relsp, figsize=(5.8, 3), yscale='log', fname='relsp.pdf', dpi=720,
             label=r'$\varepsilon=' + latex_float(epsilon) + '$',
             axhline_y=epsilon, ylabel='$s$', xlabel='index')

    # Check for bidimensional representation selection problems
    levels_cp = list(levels)
    states_cp = list(states)
    log = open('log.txt', 'a')
    log.write('\n\nlevels epsilon: ' + str(epsilon))
    for i in range(len(levels_cp)):
        if levels_cp[i].size > 2:
            local_relsp = np.diff(levels_cp[i]) / avgSpacing
            log.write('\nInfo: Found ' + str(levels_cp[i].size) + ' levels ' +
                      'in the bidimensional representation with: \nenergy: ' +
                      str(levels_cp[i]) + '\ndelta: ' +
                      str(np.diff(levels_cp[i])) + '\nrelsp: ' +
                      str(local_relsp))
            # Try to fix the problem
            if levels_cp[i].size > 3:
                log.write('\nError: Cannot choose where to split!')
                raise RuntimeError('Cannot choose where to split!')
            elif local_relsp[0] == local_relsp[1]:
                log.write('\nWarning: 3 consecutive levels with identical ' +
                          'relative spacings')
                # log.write('\nket: ' + str(states_cp[i]))
                n2 = np.array([states_cp[i][j][1] for j in range(3)])
                log.write('\nn2: ' + str(n2))
                # Find the dominant parity
                unique, counts = np.unique(n2 % 2, return_counts=True)
                log.write('\nDominant parity: ' +
                          ('odd' if unique[np.argmax(counts)] else 'even'))
                # Find the current position
                j = [np.array_equal(levels_cp[i], k)
                     for k in levels].index(True)
                # Select the levels with different parity for the bidimensional
                # representation
                dominant = n2 % 2 == unique[np.argmax(counts)]
                different = n2 % 2 != unique[np.argmax(counts)]
                # Bidimensional representation levels
                bd_l = [levels_cp[i][dominant][0]]
                # Bidimensional representation states
                bd_st = [states_cp[i][dominant][0]]
                if counts[0] < 3 and counts[1] < 3:
                    bd_l.append(levels_cp[i][different][0])
                    bd_st.append(states_cp[i][different][0])
                else:
                    logging.warning('3 consecutive quantum numbers with ' +
                                    'the same parity!')
                    bd_l.append(levels_cp[i][dominant][2])
                    bd_st.append(states_cp[i][dominant][2])
                # Unidimensional representation levels
                u_l = [levels_cp[i][dominant][1]]
                # Unidimensional representation states
                u_st = [states_cp[i][dominant][1]]

                levels[j:j] = [np.array(bd_l), np.array(u_l)]
                states[j:j] = [np.array(bd_st), np.array(u_st)]
                del levels[j + 2]
                del states[j + 2]

                log.write('\nresult: ' + str(levels[j]) + str(levels[j + 1]) +
                          '\nwith: ' + str(states[j]) + str(states[j + 1]))
            else:
                # Find the current position
                j = [np.array_equal(levels_cp[i], k)
                     for k in levels].index(True)
                # Split at the maximum relative spacing
                levels[j:j] = np.split(levels_cp[i], np.where(
                    local_relsp == local_relsp.max())[0] + 1)
                states[j:j] = np.split(states_cp[i], np.where(
                    local_relsp == local_relsp.max())[0] + 1)
                del levels[j + 2]
                del states[j + 2]
                log.write('\nresult: ' + str(levels[j]) + str(levels[j + 1]))

    k = 0
    for i in range(len(levels)):
        for j in range(levels[i].size):
            if return_colors:
                colormap[i + j + k] = colors[i % len(colors)]
            if levels[i].size > 1:  # degenerate subspace -> rebde
                ir_reps[i + j + k] = 2
            else:
                if states[i][0][1] % 2:     # n2 odd -> reuna
                    ir_reps[i + j + k] = 1
                else:                       # n2 even -> reuns
                    ir_reps[i + j + k] = 0
        k += levels[i].size - 1

    log.close()
    if return_colors:
        return ir_reps, colormap
    return ir_reps
