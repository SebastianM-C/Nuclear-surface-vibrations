#!/usr/bin/env python

import os
import numpy as np
# from timeit import default_timer as timer

import eigensystem
from tools import cd, format_float
from custom_parser import parse


def optional_color(element, color):
    """Add optional color to the element"""
    return '{\\color{' + color + '}' + element + '}' if color else element


def eigv_string(eigv, max_display, color='', k=-1, sep=',\\ '):
    """Build the eigenvector string
    Add \color{} to the eigenvector (optional)
    Add \mathbf{} to the k-th element (optional)"""

    v = ('\t\t{\\color{' + color + '}' if color else '') + \
        '\\begin{pmatrix}\n\t\t\t'
    # Limit the number of displayed eigenvector elements
    eigv_size = eigv.size if eigv.size < max_display else int(max_display / 4)
    v += ' \\\\ '.join(('\\mathbf{' + format_float(eigv[j]) + '}' if k == j
                        else format_float(eigv[j]))
                       for j in range(eigv_size))
    v += '\n\t\t\\end{pmatrix}' + ('}' if color else '') + sep + '\n'
    return v


def h_elem(elem, color='', skip_zero=False):
    """Build Hamiltonian element string
    Optionally add color and skip zeros"""
    return (optional_color(format_float(elem), color)) \
        if (not skip_zero) or elem != 0 else ''


def ket(n1, n2, color='', bf=False):
    """Build \ket{} with optional color and optional bold face"""
    k = '\\ket{' + str(n1) + '\, ' + str(n2) + '}'
    k = '\\mathbf{' + k + '}' if bf else k
    return optional_color(k, color)


def write_file(filename, nn, H, E, eigenvalues, eigenvectors, index, c_max,
               ir_reps, colors, colormap, dgc):
    """Write the LaTeX file"""
    max_display = 150   # maximum number of elements to display
    with open(filename, "w") as f:
        # Set paper size according to the imput
        if nn < 45:
            # Set paper size and number of eigenvectors on a single row
            if nn < 15:
                paper_size = str(4)
                no_eigv = 5
            if nn >= 15:
                paper_size = str(2)
                no_eigv = 10
            if nn >= 21:
                paper_size = str(1)
                no_eigv = 15
            if nn >= 28:
                paper_size = str(0)
                no_eigv = 20
            if dgc:
                no_eigv *= 2

            f.write("\\documentclass[a" + paper_size +
                    "paper,12pt,landscape]{article}\n\n")
        else:
            paper_w = int(2**(nn - 53)) * 50
            paper_h = int(2**(nn - 54)) * 50
            if nn < 55:
                paper_w = 130
                paper_h = 80
            # Limit paper size
            if paper_w > 500:
                paper_w = 200 if nn > max_display else 250
            if paper_h > 500:
                paper_h = 110
            f.write("\\documentclass[12pt,landscape]{article}\n")
            f.write("\\setlength{\paperwidth}{" + str(paper_w) + 'cm' + "}\n")
            f.write("\\setlength{\paperheight}{" + str(paper_h) + 'cm' + "}\n")
            # Use ~6cm for an eigenvector
            no_eigv = int(paper_w / 6) \
                if nn < max_display else int(max_display / 5)
        # LaTeX packages
        f.write("\\usepackage[margin=1cm]{geometry}\n")
        f.write("\\usepackage{amsmath,amsfonts,amssymb}\n")
        f.write("\\usepackage{physics}")
        f.write("\\usepackage[dvipsnames]{xcolor}\n")
        # Additional options
        f.write("\\setcounter{MaxMatrixCols}{" +
                (str(nn + 10) if nn < max_display else str(max_display)) +
                "}\n")
        f.write("\\allowdisplaybreaks\n\n")
        # The document begins here
        f.write("\\begin{document}\n\n")

        f.write("\tEigenvectors:\n")
        f.write("\t\\begin{align*}\n")

        for i in range(nn):
            # Limit the number of displayed eigenvectors
            if i == max_display:
                f.write("\t\t\cdots")
                break
            f.write(('\\\\' if i % no_eigv == 0 and i != 0 else '') +
                    '\t\tv_{' + str(i + 1) + '} ' +
                    ('&' if i % no_eigv == 0 else '') +
                    '=\n')

            # The energy corresponding to the eigenvector is given by the
            # (i+1)-th eigenvalue
            # Test if the eigenvectors correspond to the eigenvalues
            # in the degenerate case
            if dgc and \
                    np.where(np.abs(H[c_max[i]] - E[i]) < 1e-6)[0].size == 0:
                print('Warning: v' + str(i + 1) +
                      ' does not correspond to the eigenvalue ' + str(E[i]))

            if i != nn - 1:
                f.write(eigv_string(eigenvectors[i], max_display, colormap[i],
                                    c_max[i]))
            else:   # now new line after the last eigenvector
                f.write(eigv_string(eigenvectors[i], max_display,
                                    colormap[i], c_max[i], sep=''))

        f.write("\t\\end{align*}\n")
        # Skip displaying the Hamiltonian if the size is too large
        if nn < max_display - 50:
            f.write("\tHamiltonian:\n")
            f.write("\t\\[\n\tH=\n\t\\begin{pmatrix}\n")
            for i in range(nn):
                f.write("\t\t")
                # Add colors on the diagonal and remove zeros in the degenerate
                # case
                f.write(' & '.join(
                    h_elem(H[i][j]) if not dgc else
                    h_elem(H[i][j],
                           color=colors[int(round(H[i][j]))] if i == j else '',
                           skip_zero=True if not (i == j) else False)
                        for j in range(H[i].size)))
                f.write('\\\\\n')
            f.write("\t\\end{pmatrix}\n\t\\]\n")

        f.write("\tEigenvalues:\n")
        # Limit the number of eigenvalues displayed
        eigenvalues = ', '.join(eigenvalues.split(', ')[:int(max_display / 2)])
        f.write("\t\\[\lambda = " + eigenvalues + '\\]\n')

        f.write("\tStates: \n\t\\[")
        # Limit the number of displayed elements
        for i in index[:max_display]:
            f.write(ket(i[0], i[1]))
        f.write("\\]\n")

        f.write("\tOrdering: \n\t\\[")
        for i in range(nn):
            # Limit the number of displayed elements
            if i == max_display:
                f.write("\t\t\cdots")
                break
            n1 = index[c_max[i]][0]
            n2 = index[c_max[i]][1]
            f.write(ket(n1, n2, colormap[i]) + '\, ' +
                    # add newline for shorter lines
                    ('\n\t' if i % 4 == 0 and i else '')
                    )
        f.write("\\]\n")

        f.write("\tirreducible representations: \n\t\\[")
        for i in range(nn):
            # Limit the number of displayed elements
            if i == max_display:
                f.write("\t\t\cdots")
                break
            n1 = index[c_max[i]][0]
            n2 = index[c_max[i]][1]
            f.write(ket(n1, n2, colors[ir_reps[i]], ir_reps[i] == 2))
        f.write("\\]\n")

        # Check for duplicate states
        unique, counts = np.unique(c_max, return_counts=True)
        if np.any(counts > 1):
            # print("Warning: Duplicates")
            f.write("\tDuplicates: \n\t\\[")
            for i in (index[unique[counts > 1]])[:int(max_display * 1.2)]:
                f.write(ket(i[0], i[1]))
            f.write("\\]\n")

        # End of the document
        f.write("\n\n\\end{document}")


def get(b, d, n):
    """Get the results for the given parameters"""
    print("Running with: B = " + str(b) + " D = " + str(d) + " N = " + str(n))
    nn = int(n * (n + 1) / 2)

    cd(b, d, n)

    E, eigenvectors, index, c_max, H = \
        eigensystem.get(return_eigv=True, return_index=True,
                        return_cmax=True, return_H=True)

    # All available colors
    colors = ('black', 'red', 'teal', 'blue', 'orange', 'olive',
              'magenta', 'cyan', 'Brown', 'Goldenrod', 'Green', 'Violet')

    ir_reps, colormap = eigensystem.levels(E, index[c_max], colors=colors)

    # Build eigenvalue string
    # If the colormap element corresponding to the i-th eigenvalue is empty,
    # skip adding \color
    eigenvalues = ', '.join(optional_color(format_float(E[i]), colormap[i])
                            for i in range(E.size))

    return nn, H, E, eigenvalues, eigenvectors, index, c_max, ir_reps, \
        colors, colormap


def main(B, D, N):
    filename = 'results B' + str(b) + ' D' + str(d) + ' N' + str(n) + '.tex'
    write_file(filename, *get(b, d, n),
               dgc=(b == 0 and d == 0)  # degenerate case
               )
    os.chdir("../../Scripts")
    print("Done")


if __name__ == '__main__':
    B, D, N = parse()
    for b in B:
        for d in D:
            for n in N:
                main(B, D, N)
