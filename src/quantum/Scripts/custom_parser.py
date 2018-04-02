import argparse
import numpy as np


def parse(advanced=False, select=False, hist_bin=False, max_e=False,
          e_plot=False, s_plot=False):
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=np.float64, nargs='+', default=[0.2],
                        help='Hamiltonian B parameter')
    parser.add_argument('-d', type=np.float64, nargs='+', default=[0.4],
                        help='Hamiltonian D parameter')
    parser.add_argument('-n', type=np.int64, nargs='+', default=[4],
                        help='Diagonalisation basis size')
    parser.add_argument('-dn', '--delta_n', type=np.int64, nargs=1,
                        default=20,
                        help='Size difference between diagonalization bases')
    parser.add_argument('-st_eps', '--stability_epsilon', type=np.float64,
                        default=1e-9,
                        help='Maximum difference between two stable levels' +
                        'when the diagonalization basis size is increased by' +
                        'delta_n')
    parser.add_argument('-l_eps', '--levels_epsilon', type=np.float64,
                        default=1e-8,
                        help='Minimum difference between two consecutive' +
                        'levels from one of the irreducible' +
                        'unidimensional representations')
    parser.add_argument('-r', '--reselect', action='store_false',
                        default=True, help='Specify whether to reselect the' +
                        'irreducible representations or not')
    parser.add_argument('-c', '--cut', type=np.float64,
                        default=0,
                        help='Factor by which to reduce the number of stable' +
                        'levels when checking the convergence of the results')
    parser.add_argument('-bin', '--bin_size', type=np.float64,
                        default=0.25, help='P(S) histogram bin size')
    parser.add_argument('-max_e', '--max_energy', nargs='+', type=np.float64,
                        default=[0], help='Use only the levels with energy' +
                        ' up to the specified one')
    parser.add_argument('-ep', '--energy_plot', action='store_true',
                        default=False, help='Plot alpha as a function of' +
                        'energy')
    parser.add_argument('-sp', '--small_plot', action='store_true',
                        default=False, help='Generate smaller plots')

    # args = parser.parse_args(input().split())
    args = parser.parse_args()

    # Hamiltonian parameters
    B = args.b
    D = args.d
    N = args.n
    delta_n = args.delta_n
    st_epsilon = args.stability_epsilon
    lvl_epsilon = args.levels_epsilon
    reselect = args.reselect
    cut = args.cut
    bin_size = args.bin_size
    max_energy = args.max_energy
    energy_plot = args.energy_plot
    small_plot = args.small_plot

    arguments = (B, D, N)

    if advanced:
        arguments += (delta_n, st_epsilon, lvl_epsilon)
    if select:
        arguments += (reselect, cut)
    if hist_bin:
        arguments += (bin_size, )
    if max_e:
        arguments += (max_energy, )
    if e_plot:
        arguments += (energy_plot, )
    if s_plot:
        arguments += (small_plot, )

    return arguments
