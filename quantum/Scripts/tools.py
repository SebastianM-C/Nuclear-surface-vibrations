#!/usr/bin/env python

import os
import re
from fnmatch import fnmatch


def get_input(path=''):
    """Return the input parameters (b, d, n) of the given path
    or the current path if no argument is provided"""
    if not path:
        path = os.getcwd()
    regex = r"""B(0.[0-9]+)+ D(0.[0-9]+)+ N([0-9]+)"""
    return [float(i) for i in
            re.compile(regex).search(path).group(1, 2, 3)]


def exists(b, d, n):
    """Check if the folder specified by the given parameters exists"""
    return os.path.isdir("../Output/Quantum/B" +
                         str(b) + " D" + str(d) + " N" + str(n))


def clean_dir(directory):
    """If the directory doesn't exists create it, else clean it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for the_file in os.listdir(directory):
            file_path = os.path.join(directory, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


def cd(b, d, n):
    """Try to cd to the given path. In case of an error go back to ../../Scripts
    and try again (maybe the last run had an error or
    the script did not reach the end)."""

    # Check if already there
    try:
        if [b, d, n] == get_input():
            # print('Already there:', os.getcwd())
            return
        else:
            os.chdir("../../../Scripts")
            # print('Went back to Scripts')
    except AttributeError as e:
        pass
    # The script should be in the Scripts directory
    if exists(b, d, n):
        os.chdir('../Output/Quantum/B' + str(b) + ' D' + str(d) + ' N' + str(n))
        # print('Succes: ', os.getcwd())
    else:
        print('The specified directory does not exist!\n',
              'Now in: ' + os.getcwd())


def format_float(n):
    """Converts the number to int if possible"""
    return ('{:n}' if n == int(n) else '{:.8g}').format(n)


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if base != 1:
            return r"10^{{{0}}}".format(int(exponent))
        else:
            return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def find(pattern, path):
    """Find all the files with the given name in the given path"""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
