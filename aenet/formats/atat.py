#!/usr/bin/env python

"""
ATAT (Alloy Theoretic Automated Toolkit) structure format.

"""

import numpy as np
import sys

from ..geometry import AtomicStructure
from .parser_abc import ParserABC
from .. import util
from ..exceptions import ArgumentError

__author__ = "Alexander Urban"
__date__ = "2015-05-01"


class ATATParser(ParserABC):
    def __init__(self):
        self.name = 'atat'
        self.description = "ATAT's structure format"
        self.extensions = []
        self.default_file_names = ['str.out']

    def read(self, infile, **kwargs):
        """
        Parse atomic structure file in ATAT's structure (str.out) format.

        Arguments:
          infile   name of the input file

        Returns:
          instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        line = f.readline().split()
        if len(line) == 3:
            basis = [[float(el) for el in line]]
            basis.append([float(el) for el in f.readline().split()])
            basis.append([float(el) for el in f.readline().split()])
            basis = np.array(basis)
        else:
            basis = util.cellmatrix_from_params(*[float(el) for el in line])

        cell = []
        for i in range(3):
            cell.append([float(el) for el in f.readline().split()])
        cell = np.array(cell)

        coords = []
        types = []
        line = f.readline()
        while line:
            coords.append([float(el) for el in line.split()[:3]])
            types.append(line.split()[3])
            line = f.readline()
        coords = np.array(coords).dot(basis)

        if close_file:
            f.close()

        avec = cell.dot(basis)

        struc = AtomicStructure(coords, types=types, avec=avec,
                                fractional=False)
        struc.add_comment("Converted from ATAT structure format")

        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=-1, **kwargs):
        """
        Write atomic structure in ATAT's structure (str.out) format.

        Arguments:
          struc       instance of the AtomicStructure class
          outfile     name of the output file; if None, the contents
                      will be written to stdout
          frame       For trajectories, frame to be used.

        """

        if not struc.pbc:
            raise ArgumentError(
                "Error: the ATAT's str.out format requires a periodic lattice")

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if outfile:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        f.write("1.0 1.0 1.0 90.0 90.0 90.0\n")
        for i in range(3):
            f.write((3*"{:15.8f} ").format(*struc.avec[frame][i]) + "\n")
        for i in range(struc.natoms):
            f.write((3*"{:15.8f} ").format(*struc.coords[frame][i]))
            f.write("{}".format(struc.types[i]) + "\n")

        if outfile:
            f.close()
